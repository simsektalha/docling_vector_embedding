import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


