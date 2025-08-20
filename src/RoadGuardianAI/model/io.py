import joblib
from pathlib import Path
from typing import Any

def save_preprocessor(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_preprocessor(path: str):
    return joblib.load(path)
