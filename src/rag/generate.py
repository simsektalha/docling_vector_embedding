from typing import Dict, List

from src.common.config import load_config
from src.common.types import SearchResult
from src.rag.assemble_prompt import assemble


def answer(question: str, contexts: List[SearchResult], cfg_path: str) -> Dict:
    cfg = load_config(cfg_path)
    prompt = assemble(contexts, question, max_context_chars=min(cfg["rag"].get("max_context_tokens", 4000) * 4, 16000))
    try:
        # Use external OpenAI-compatible endpoint when configured
        base_url = cfg.get("llm", {}).get("base_url")
        api_key = cfg.get("llm", {}).get("api_key")
        model = cfg.get("llm", {}).get("model") or cfg["rag"].get("llm_model")
        if base_url:
            import httpx

            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "stream": False,
                },
                timeout=cfg.get("llm", {}).get("timeout_s", 60),
            )
            resp.raise_for_status()
            j = resp.json()
            text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            completion = client.chat.completions.create(
                model=model or cfg["rag"]["llm_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = completion.choices[0].message.content
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


