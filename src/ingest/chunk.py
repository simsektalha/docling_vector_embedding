from typing import List

import tiktoken

from src.common.types import Chunk, DocumentConversion


def _split_tokens(text: str, max_tokens: int, overlap_tokens: int, encoder) -> List[str]:
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def hierarchical_chunk(conversion: DocumentConversion, max_tokens: int, overlap_tokens: int) -> List[Chunk]:
    encoder = tiktoken.get_encoding("cl100k_base")
    result: List[Chunk] = []
    chunk_index = 0
    for section in conversion.sections:
        pieces = _split_tokens(section.text, max_tokens, overlap_tokens, encoder)
        offset = 0
        for piece in pieces:
            start = offset
            end = offset + len(piece)
            result.append(
                Chunk(
                    doc_id=conversion.doc_id,
                    chunk_index=chunk_index,
                    text=piece,
                    char_span=(start, end),
                    section_path=section.section_path,
                    page_numbers=section.page_numbers,
                    metadata={},
                )
            )
            chunk_index += 1
            offset = end
    return result


def token_chunk(plain_text: str, doc_id: str, max_tokens: int, overlap_tokens: int) -> List[Chunk]:
    encoder = tiktoken.get_encoding("cl100k_base")
    pieces = _split_tokens(plain_text, max_tokens, overlap_tokens, encoder)
    result: List[Chunk] = []
    offset = 0
    for idx, piece in enumerate(pieces):
        start = offset
        end = offset + len(piece)
        result.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=idx,
                text=piece,
                char_span=(start, end),
                section_path=None,
                page_numbers=[],
                metadata={},
            )
        )
        offset = end
    return result


def chunk_document(conversion: DocumentConversion, strategy: str, max_tokens: int, overlap_tokens: int) -> List[Chunk]:
    if strategy == "hierarchical" and conversion.sections:
        return hierarchical_chunk(conversion, max_tokens, overlap_tokens)
    joined = "\n\n".join(s.text for s in conversion.sections)
    return token_chunk(joined, conversion.doc_id, max_tokens, overlap_tokens)


