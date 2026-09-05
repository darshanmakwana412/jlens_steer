import argparse
import sys

from assets import ASSETS, ROOT
from hub import auth_headers, fetch, human_bytes, list_remote_files


def log(message: str) -> None:
    print(message, flush=True)


def verify(files) -> list:
    failed = []
    for remote in files:
        ok = remote.verified()
        log(f"  {'ok  ' if ok else 'FAIL'}     {remote.name}")
        if not ok:
            failed.append(remote)
    return failed


def parse_args():
    parser = argparse.ArgumentParser(description="Download project assets into the repo.")
    parser.add_argument("assets", nargs="*", choices=list(ASSETS), metavar="ASSET")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = auth_headers()
    if not headers:
        log("no HF token found; downloads will be rate limited")

    failed = []
    for name in args.assets or list(ASSETS):
        asset = ASSETS[name]
        files = list_remote_files(asset)
        total = sum(remote.size for remote in files)
        log(
            f"\n{name}: {asset.repo_id} -> {asset.local_dir.relative_to(ROOT)}"
            f" ({human_bytes(total)}, {len(files)} files)"
        )
        if args.verify:
            failed += verify(files)
        else:
            for remote in files:
                fetch(remote, headers, log)

    if args.verify:
        log(f"\n{len(failed)} file(s) failed" if failed else "\nall files verified")
        return 1 if failed else 0

    log("\nall assets present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
