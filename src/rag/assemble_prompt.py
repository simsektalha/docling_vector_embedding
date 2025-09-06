from typing import List

from src.common.types import SearchResult


def assemble(prompt_contexts: List[SearchResult], question: str, max_context_chars: int = 8000) -> str:
    parts = []
    used = 0
    for r in prompt_contexts:
        header = f"Source: {r.metadata.get('source_path','')} | Section: {r.metadata.get('section_path','')} | Score: {r.score:.3f}\n"
        body = r.text
        if used + len(header) + len(body) > max_context_chars:
            space = max_context_chars - used - len(header)
            if space <= 0:
                break
            body = body[:space]
        parts.append(header + body)
        used += len(header) + len(body)
        if used >= max_context_chars:
            break
    context = "\n\n".join(parts)
    return f"You are a helpful assistant. Use only the context below. Cite sources.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


