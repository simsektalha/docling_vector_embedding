from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st

from src.common.config import load_config
from src.common.types import SearchResult
from src.rag.retrieve import retrieve
from src.rag.generate import answer


def _get_cfg_path() -> str:
    return os.getenv("CONFIG_PATH", "configs/default.yaml")


def _render_sources(sources: List[Dict]) -> None:
    for s in sources:
        st.caption(f"{s.get('source_path','')} · {s.get('section_path','')} · score={s.get('score',0.0):.3f}")


def main() -> None:
    st.set_page_config(page_title="Doc RAG Chat", page_icon="📄", layout="centered")
    cfg_path = _get_cfg_path()
    cfg = load_config(cfg_path)

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("Postgres DSN", cfg.get("pg", {}).get("dsn", ""), disabled=True)
        st.text_input("Embeddings Base URL", cfg.get("embedding", {}).get("base_url", ""), disabled=True)
        st.text_input("LLM Base URL", cfg.get("llm", {}).get("base_url", ""), disabled=True)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=cfg.get("retrieval", {}).get("top_k", 5))

    st.title("Chat with your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role, content}

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question about your docs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                contexts: List[SearchResult] = retrieve(prompt, cfg_path, top_k, None)
                res = answer(prompt, contexts, cfg_path)
                st.markdown(res["answer"] or "[No answer]")
                _render_sources(res.get("sources", []))
        st.session_state.messages.append({"role": "assistant", "content": res["answer"]})


if __name__ == "__main__":
    main()


