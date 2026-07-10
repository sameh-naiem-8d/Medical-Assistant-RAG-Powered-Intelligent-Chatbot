from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root / "artifacts_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest:
        dest = root / item["expected_destination"]
        if not dest.exists():
            failures.append(f"missing {dest}")
            continue
        size = dest.stat().st_size
        digest = sha256_file(dest)
        if size != int(item["size_bytes"]):
            failures.append(f"size mismatch {dest}: {size} != {item['size_bytes']}")
        if digest.lower() != str(item["sha256"]).lower():
            failures.append(f"sha256 mismatch {dest}")
    if failures:
        print("Artifact verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Artifact verification passed for {len(manifest)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
