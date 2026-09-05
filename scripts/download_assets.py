"""Download every external asset this project depends on into the repo.

Everything lands under models/ and artifacts/ (never the global HF cache), so
the repo is self-contained.

    python scripts/download_assets.py             # all assets
    python scripts/download_assets.py base jlens  # a subset
    python scripts/download_assets.py --verify    # re-check hashes, download nothing

Why not plain `snapshot_download`? On this machine long TLS streams to the HF
CDN die partway through with "[SSL] record layer failure", and hf_hub discards
the partial file, so retries never make progress. Ranged requests of a few tens
of MB do succeed. So we fetch each file as a sequence of chunks written at fixed
offsets into a preallocated `.part`, record which chunks landed, retry only the
failed ones, and verify the LFS sha256 before publishing the file. Interrupt it
at any point and re-running picks up where it stopped.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

CHUNK = 16 << 20  # 16 MiB: comfortably below where the TLS streams start dying
WORKERS = 4
ATTEMPTS = 25
ROUNDS = 8  # outer passes, so one unlucky chunk cannot end the run
TIMEOUT = httpx.Timeout(30.0, read=120.0)

# The abliterated repo carries two redundant weight sets: the fp32 shards its
# model.safetensors.index.json points at, and a leftover single-file bf16
# model.safetensors. We take the indexed shards and skip the leftover, which
# would otherwise shadow the index at load time.
ASSETS = {
    "base": dict(
        repo_id="Qwen/Qwen3-1.7B",
        local_dir=MODELS / "Qwen3-1.7B",
    ),
    "abliterated": dict(
        repo_id="mlabonne/Qwen3-1.7B-abliterated",
        local_dir=MODELS / "Qwen3-1.7B-abliterated",
        exclude=["model.safetensors"],
    ),
    # 58 GB repo covering many models; we only need the Qwen3-1.7B lens.
    "jlens": dict(
        repo_id="neuronpedia/jacobian-lens",
        local_dir=ARTIFACTS / "jacobian-lens",
        include=["README.md", "qwen3-1.7b/"],
    ),
    "steering_vecs": dict(
        repo_id="science-of-finetuning/steering-vecs-qwen3_1_7B",
        local_dir=ARTIFACTS / "steering-vecs-qwen3_1_7B",
    ),
}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def sha256_file(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(buf):
            h.update(block)
    return h.hexdigest()


class FileSpec:
    """One remote file: where it lives, how big it is, what it should hash to."""

    def __init__(self, repo_id, revision, rfilename, size, sha256, dest):
        self.repo_id = repo_id
        self.revision = revision
        self.rfilename = rfilename
        self.size = size
        self.sha256 = sha256  # None for non-LFS files; size is the only check
        self.dest = dest

    @property
    def url(self):
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.rfilename}"

    @property
    def part(self):
        return self.dest.with_suffix(self.dest.suffix + ".part")

    @property
    def state(self):
        return self.dest.with_suffix(self.dest.suffix + ".part.json")

    def is_done(self, verify=False):
        if not self.dest.exists() or self.dest.stat().st_size != self.size:
            return False
        if verify and self.sha256:
            return sha256_file(self.dest) == self.sha256
        return True


def list_files(repo_id, include=None, exclude=None):
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    out = []
    for sib in info.siblings:
        name = sib.rfilename
        if include and not any(name == p or name.startswith(p) for p in include):
            continue
        if exclude and name in exclude:
            continue
        if name in (".gitattributes", ".DS_Store") or name.endswith("/.DS_Store"):
            continue
        out.append((name, sib.size or 0, getattr(sib.lfs, "sha256", None) if sib.lfs else None))
    return sorted(out), info.sha


def fetch_chunks(spec, headers):
    """Fill spec.part with all missing chunks, retrying each independently."""
    n_chunks = max(1, -(-spec.size // CHUNK))
    done = set()
    if spec.part.exists() and spec.state.exists():
        try:
            saved = json.loads(spec.state.read_text())
            if saved.get("size") == spec.size and saved.get("etag") == spec.sha256:
                done = set(saved.get("chunks", []))
        except (ValueError, OSError):
            done = set()

    spec.part.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(spec.part, os.O_RDWR | os.O_CREAT)
    try:
        if os.fstat(fd).st_size != spec.size:
            os.ftruncate(fd, spec.size)
            done = set()

        todo = [i for i in range(n_chunks) if i not in done]
        if not todo:
            return
        state_lock = threading.Lock()
        transferred = [0]

        def save_state():
            spec.state.write_text(
                json.dumps({"size": spec.size, "etag": spec.sha256, "chunks": sorted(done)})
            )

        def do_chunk(i):
            start = i * CHUNK
            end = min(start + CHUNK, spec.size) - 1
            last_err = None
            for attempt in range(ATTEMPTS):
                try:
                    # A fresh client per attempt: a connection that just failed
                    # mid-stream is not worth reusing.
                    with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
                        r = client.get(
                            spec.url, headers={**headers, "Range": f"bytes={start}-{end}"}
                        )
                        r.raise_for_status()
                        data = r.content
                    if len(data) != end - start + 1:
                        raise OSError(f"short chunk: {len(data)} != {end - start + 1}")
                    os.pwrite(fd, data, start)
                    with state_lock:
                        done.add(i)
                        transferred[0] += len(data)
                        save_state()
                        pct = 100 * len(done) / n_chunks
                        log(
                            f"    {spec.rfilename}  {pct:5.1f}%  "
                            f"({len(done)}/{n_chunks} chunks, {human(transferred[0])} this run)"
                        )
                    return
                except Exception as exc:  # noqa: BLE001 - transport errors vary
                    last_err = exc
                    time.sleep(min(10, 1.5**attempt))
            raise RuntimeError(f"chunk {i} of {spec.rfilename} failed: {last_err}")

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(do_chunk, i) for i in todo]
            for fut in as_completed(futures):
                fut.result()
    finally:
        os.close(fd)


def download(spec, headers):
    if spec.is_done():
        log(f"  skip  {spec.rfilename}  ({human(spec.size)}, already present)")
        return
    log(f"  get   {spec.rfilename}  ({human(spec.size)})")
    for round_no in range(1, ROUNDS + 1):
        try:
            fetch_chunks(spec, headers)
            break
        except RuntimeError as exc:
            if round_no == ROUNDS:
                raise
            log(f"  retry {spec.rfilename}  round {round_no}: {exc}")
            time.sleep(5)

    if spec.sha256:
        got = sha256_file(spec.part)
        if got != spec.sha256:
            spec.part.unlink(missing_ok=True)
            spec.state.unlink(missing_ok=True)
            raise SystemExit(
                f"sha256 mismatch for {spec.rfilename}: got {got}, expected {spec.sha256}. "
                "Removed the partial file; re-run to try again."
            )
        log(f"  ok    {spec.rfilename}  sha256 verified")
    spec.part.replace(spec.dest)
    spec.state.unlink(missing_ok=True)


def resolve(name):
    cfg = dict(ASSETS[name])
    local_dir = cfg["local_dir"]
    files, revision = list_files(cfg["repo_id"], cfg.get("include"), cfg.get("exclude"))
    return [
        FileSpec(cfg["repo_id"], revision, rf, size, sha, local_dir / rf)
        for rf, size, sha in files
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assets", nargs="*", default=None, help=f"subset of {list(ASSETS)}")
    ap.add_argument("--verify", action="store_true", help="re-hash local files, download nothing")
    args = ap.parse_args()

    names = args.assets or list(ASSETS)
    unknown = [n for n in names if n not in ASSETS]
    if unknown:
        sys.exit(f"unknown asset(s) {unknown}; choose from {list(ASSETS)}")

    token = get_token() or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if not token:
        log("note: no HF token found; anonymous downloads are rate-limited")

    failures = 0
    for name in names:
        specs = resolve(name)
        total = sum(s.size for s in specs)
        log(f"\n=== {name}: {specs[0].repo_id} -> "
            f"{ASSETS[name]['local_dir'].relative_to(ROOT)}  ({human(total)}, {len(specs)} files)")
        for spec in specs:
            if args.verify:
                ok = spec.is_done(verify=True)
                failures += not ok
                log(f"  {'ok  ' if ok else 'FAIL'}  {spec.rfilename}")
            else:
                download(spec, headers)

    if args.verify:
        log(f"\n{'all files verified' if not failures else f'{failures} file(s) failed'}")
        sys.exit(1 if failures else 0)
    log("\nall assets present")


if __name__ == "__main__":
    main()
