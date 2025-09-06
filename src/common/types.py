from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DiscoveredFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass
class SectionText:
    section_path: str
    page_numbers: List[int]
    text: str


@dataclass
class DocumentConversion:
    doc_id: str
    title: Optional[str]
    author: Optional[str]
    created_at: Optional[str]
    modified_at: Optional[str]
    language: Optional[str]
    markdown: Optional[str]
    source_path: Optional[str]
    dl_doc: Optional[object]
    sections: List[SectionText]


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_span: Tuple[int, int]
    section_path: Optional[str]
    page_numbers: List[int]
    metadata: Dict[str, Any]


@dataclass
class EmbeddingRecord:
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


