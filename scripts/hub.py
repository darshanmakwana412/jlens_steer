import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token

CHUNK_BYTES = 16 << 20
MAX_WORKERS = 4
CHUNK_ATTEMPTS = 25
FILE_ROUNDS = 8
ROUND_PAUSE = 5.0
TIMEOUT = httpx.Timeout(30.0, read=120.0)


class DownloadError(RuntimeError):
    pass


def human_bytes(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def auth_headers() -> dict[str, str]:
    token = get_token() or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


@dataclass(frozen=True)
class RemoteFile:
    repo_id: str
    revision: str
    name: str
    size: int
    sha256: str | None
    dest: Path

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.name}"

    @property
    def part(self) -> Path:
        return self.dest.with_name(self.dest.name + ".part")

    @property
    def state(self) -> Path:
        return self.dest.with_name(self.dest.name + ".part.json")

    @property
    def n_chunks(self) -> int:
        return max(1, -(-self.size // CHUNK_BYTES))

    def chunk_range(self, index: int) -> tuple[int, int]:
        start = index * CHUNK_BYTES
        return start, min(start + CHUNK_BYTES, self.size) - 1

    def present(self) -> bool:
        return self.dest.exists() and self.dest.stat().st_size == self.size

    def verified(self) -> bool:
        if not self.present():
            return False
        return not self.sha256 or sha256_file(self.dest) == self.sha256


def list_remote_files(asset) -> list[RemoteFile]:
    info = HfApi().model_info(asset.repo_id, files_metadata=True)
    files = [
        RemoteFile(
            repo_id=asset.repo_id,
            revision=info.sha,
            name=sibling.rfilename,
            size=sibling.size or 0,
            sha256=sibling.lfs.sha256 if sibling.lfs else None,
            dest=asset.local_dir / sibling.rfilename,
        )
        for sibling in info.siblings
        if asset.wants(sibling.rfilename)
    ]
    return sorted(files, key=lambda remote: remote.name)


def _silent(message: str) -> None:
    return None


def _load_progress(remote: RemoteFile) -> set[int]:
    if not (remote.part.exists() and remote.state.exists()):
        return set()
    try:
        saved = json.loads(remote.state.read_text())
    except (OSError, ValueError):
        return set()
    if saved.get("size") != remote.size or saved.get("sha256") != remote.sha256:
        return set()
    return set(saved.get("chunks", []))


def _save_progress(remote: RemoteFile, done: set[int]) -> None:
    remote.state.write_text(
        json.dumps({"size": remote.size, "sha256": remote.sha256, "chunks": sorted(done)})
    )


def _fetch_range(remote: RemoteFile, headers: dict[str, str], start: int, end: int) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
        response = client.get(remote.url, headers={**headers, "Range": f"bytes={start}-{end}"})
        response.raise_for_status()
        data = response.content
    expected = end - start + 1
    if len(data) != expected:
        raise OSError(f"short read: {len(data)} of {expected} bytes")
    return data


def _download_chunks(remote: RemoteFile, headers: dict[str, str], log) -> None:
    done = _load_progress(remote)
    remote.part.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(remote.part, os.O_RDWR | os.O_CREAT)
    lock = threading.Lock()
    try:
        if os.fstat(handle).st_size != remote.size:
            os.ftruncate(handle, remote.size)
            done = set()

        pending = [index for index in range(remote.n_chunks) if index not in done]
        if not pending:
            return

        def run(index: int) -> None:
            start, end = remote.chunk_range(index)
            failure = None
            for attempt in range(CHUNK_ATTEMPTS):
                try:
                    payload = _fetch_range(remote, headers, start, end)
                except Exception as error:
                    failure = error
                    time.sleep(min(10.0, 1.5**attempt))
                    continue
                os.pwrite(handle, payload, start)
                with lock:
                    done.add(index)
                    _save_progress(remote, done)
                    share = 100 * len(done) / remote.n_chunks
                    log(f"    {remote.name} {share:5.1f}% ({len(done)}/{remote.n_chunks})")
                return
            raise DownloadError(f"chunk {index} of {remote.name}: {failure}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for future in as_completed([pool.submit(run, index) for index in pending]):
                future.result()
    finally:
        os.close(handle)


def _publish(remote: RemoteFile, log) -> None:
    if remote.sha256:
        actual = sha256_file(remote.part)
        if actual != remote.sha256:
            remote.part.unlink(missing_ok=True)
            remote.state.unlink(missing_ok=True)
            raise DownloadError(
                f"{remote.name}: sha256 {actual} != {remote.sha256}; partial file discarded"
            )
        log(f"  verified {remote.name}")
    remote.part.replace(remote.dest)
    remote.state.unlink(missing_ok=True)


def fetch(remote: RemoteFile, headers: dict[str, str], log=_silent) -> None:
    if remote.present():
        log(f"  present  {remote.name} ({human_bytes(remote.size)})")
        return

    log(f"  fetching {remote.name} ({human_bytes(remote.size)})")
    for round_index in range(FILE_ROUNDS):
        try:
            _download_chunks(remote, headers, log)
            break
        except DownloadError as error:
            if round_index == FILE_ROUNDS - 1:
                raise
            log(f"  retrying {remote.name} after {error}")
            time.sleep(ROUND_PAUSE)
    _publish(remote, log)
