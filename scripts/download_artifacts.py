from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    manifest = json.loads((repo / "artifacts_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        url = item.get("download_url")
        if not url:
            print(f"No download URL configured for {item['filename']}. Use setup_from_local_archive.py for now.")
            return 2
        dst = repo / item["expected_destination"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".download")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        if tmp.stat().st_size != int(item["size_bytes"]) or sha256_file(tmp).lower() != str(item["sha256"]).lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded artifact failed verification: {item['filename']}")
        tmp.replace(dst)
    print("Download and verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
