from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=r"D:\Project Graduation\MEDBRIDGE_FINAL_AI_DELIVERY")
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    repo = Path(args.repo).resolve()
    manifest = json.loads((repo / "artifacts_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        src = archive / item["local_archive_source"]
        dst = repo / item["expected_destination"]
        if not src.exists():
            raise FileNotFoundError(f"Archive artifact not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src.name} -> {dst}")
    return subprocess.call([sys.executable, str(repo / "scripts" / "verify_artifacts.py"), "--root", str(repo)])


if __name__ == "__main__":
    raise SystemExit(main())
