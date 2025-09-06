from typing import Dict, List

from src.common.config import load_config
from src.common.types import SearchResult
from src.rag.assemble_prompt import assemble


def answer(question: str, contexts: List[SearchResult], cfg_path: str) -> Dict:
    cfg = load_config(cfg_path)
    prompt = assemble(contexts, question, max_context_chars=min(cfg["rag"].get("max_context_tokens", 4000) * 4, 16000))
    try:
        llm_provider = cfg["rag"]["llm_provider"]
        if llm_provider == "openai":
            from openai import OpenAI

            client = OpenAI()
            completion = client.chat.completions.create(
                model=cfg["rag"]["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = completion.choices[0].message.content
        else:
            # Ollama fallback (simple HTTP call)
            import httpx

            base = "http://localhost:11434"
            model = cfg["rag"]["llm_model"]
            resp = httpx.post(f"{base}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=60)
            text = resp.json().get("response", "")
    except Exception:
        text = "[LLM unavailable]"

    sources = [
        {
            "id": r.id,
            "score": r.score,
            "source_path": r.metadata.get("source_path"),
            "section_path": r.metadata.get("section_path"),
        }
        for r in contexts
    ]
    confidence = contexts[0].score if contexts else 0.0
    return {"answer": text, "sources": sources, "confidence": confidence}


