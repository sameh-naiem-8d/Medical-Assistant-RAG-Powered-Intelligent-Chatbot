from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
targets = [ROOT / "app", ROOT / "scripts", ROOT / "tests"]
checked = 0
errors: list[str] = []
for target in targets:
    if not target.exists():
        continue
    for path in target.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        checked += 1
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except Exception as exc:
            errors.append(f"{path}: {exc.__class__.__name__}: {exc}")
if errors:
    print("Syntax check failed:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)
print(f"Syntax check passed for {checked} Python files.")
