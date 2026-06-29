"""Config + artifact IO."""
import os, json
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
