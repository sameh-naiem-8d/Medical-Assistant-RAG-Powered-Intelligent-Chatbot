from __future__ import annotations

import importlib.util
import os
import sys

REQUIRED = ["fastapi", "uvicorn", "pydantic", "dotenv", "pandas", "numpy", "sklearn", "joblib", "faiss", "sentence_transformers"]
OPTIONAL = ["groq"]

missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
optional_missing = [name for name in OPTIONAL if importlib.util.find_spec(name) is None]
print(f"Python: {sys.executable}")
print(f"ARTIFACTS_DIR: {os.getenv('ARTIFACTS_DIR', 'artifacts')}")
print(f"Groq key configured: {bool(os.getenv('GROQ_API_KEYS') or os.getenv('GROQ_API_KEY'))}")
if optional_missing:
    print("Optional packages missing: " + ", ".join(optional_missing))
if missing:
    print("Required packages missing: " + ", ".join(missing))
    raise SystemExit(1)
print("Environment check passed.")
