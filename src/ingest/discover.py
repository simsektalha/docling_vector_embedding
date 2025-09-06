import fnmatch
import hashlib
import os
from pathlib import Path
from typing import Iterable, List

from src.common.types import DiscoveredFile


def _iter_files(root: str) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def _matches(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns) if patterns else True


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_files(input_dir: str, include_glob: List[str], exclude_glob: List[str], max_file_mb: int) -> List[DiscoveredFile]:
    results: List[DiscoveredFile] = []
    for p in _iter_files(input_dir):
        rel_name = p.name
        if include_glob and not _matches(rel_name, include_glob):
            continue
        if exclude_glob and _matches(rel_name, exclude_glob):
            continue
        size_bytes = p.stat().st_size
        if size_bytes > max_file_mb * 1024 * 1024:
            continue
        results.append(DiscoveredFile(path=str(p), size_bytes=size_bytes, sha256=_sha256_of_file(p)))
    return results


