import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.common.types import DocumentConversion, SectionText


def _cache_path(cache_dir: str, sha256: str) -> str:
    return str(Path(cache_dir) / f"{sha256}.json")


def _load_cache(cache_dir: str, sha256: str) -> Dict[str, Any] | None:
    path = _cache_path(cache_dir, sha256)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(cache_dir: str, sha256: str, data: Dict[str, Any]) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(_cache_path(cache_dir, sha256), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def convert_with_docling(file_path: str, sha256: str, cfg: Dict[str, Any]) -> DocumentConversion:
    cache_dir = cfg["data"]["cache_dir"]
    if cfg.get("docling", {}).get("cache_converted", True):
        cached = _load_cache(cache_dir, sha256)
        if cached:
            return _from_cached(cached)

    # Lazy import to keep optional dependency lightweight for tests
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)

    # Extract markdown + sections
    sections: List[SectionText] = []
    # Fallback: use entire document text if sections are unavailable
    try:
        # Prefer markdown export when available
        markdown = result.document.export_to_markdown() if hasattr(result.document, "export_to_markdown") else None
        plain_text = markdown or str(result.document)
        sections.append(SectionText(section_path="Document", page_numbers=[], text=plain_text))
    except Exception:
        sections.append(SectionText(section_path="Document", page_numbers=[], text=str(result)))

    dc = DocumentConversion(
        doc_id=sha256,
        title=getattr(result, "title", None),
        author=None,
        created_at=None,
        modified_at=None,
        language="en",
        sections=sections,
        markdown=markdown if 'markdown' in locals() else None,
    )

    if cfg.get("docling", {}).get("cache_converted", True):
        _save_cache(cache_dir, sha256, _to_cached(dc))
    return dc


def _to_cached(dc: DocumentConversion) -> Dict[str, Any]:
    return {
        "doc_id": dc.doc_id,
        "title": dc.title,
        "author": dc.author,
        "created_at": dc.created_at,
        "modified_at": dc.modified_at,
        "language": dc.language,
        "markdown": dc.markdown,
        "sections": [
            {"section_path": s.section_path, "page_numbers": s.page_numbers, "text": s.text}
            for s in dc.sections
        ],
    }


def _from_cached(data: Dict[str, Any]) -> DocumentConversion:
    return DocumentConversion(
        doc_id=data["doc_id"],
        title=data.get("title"),
        author=data.get("author"),
        created_at=data.get("created_at"),
        modified_at=data.get("modified_at"),
        language=data.get("language"),
        markdown=data.get("markdown"),
        sections=[
            SectionText(section_path=s["section_path"], page_numbers=s.get("page_numbers", []), text=s["text"])  # type: ignore
            for s in data.get("sections", [])
        ],
    )


