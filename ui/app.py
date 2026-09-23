import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ask My Docs - SBA", page_icon="📄", layout="centered")
st.title("Ask My Docs: SBA Loan Requirements")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        label = f"{citation['source_doc']} (effective {citation['effective_date']})"
        with st.expander(label):
            st.write(citation["chunk_text"])


def _stream_response(question: str, history: list[dict]):
    """Yield answer tokens as they arrive; stashes citations/grounded in session_state as a side effect.

    Note: by the time any of this arrives, the answer has already been fully
    generated and verified server-side (grounding_check already ran) -- this
    is a replay of a verified answer, not a live token-by-token generation."""
    response = requests.post(
        f"{API_URL}/query/stream",
        json={"question": question, "history": history},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    event_name = None
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if line == "":
            if event_name is not None:
                data = json.loads("".join(data_lines)) if data_lines else {}
                if event_name == "citations":
                    st.session_state["_last_citations"] = data.get("citations", [])
                elif event_name == "token":
                    yield data.get("text", "")
                elif event_name == "grounded":
                    st.session_state["_last_grounded"] = data.get("grounded", True)
            event_name, data_lines = None, []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            _render_citations(message.get("citations", []))
            if not message.get("grounded", True):
                st.caption("⚠️ This answer could not be fully verified against the source documents.")

question = st.chat_input("Ask a question about SBA 7(a)/504 loan requirements")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    history = [
        {"role": "human" if m["role"] == "user" else "ai", "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        answer = None
        try:
            answer = st.write_stream(_stream_response(question, history))
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

        citations = st.session_state.pop("_last_citations", [])
        grounded = st.session_state.pop("_last_grounded", True)
        if answer is not None:
            _render_citations(citations)
            if not grounded:
                st.caption("⚠️ This answer could not be fully verified against the source documents.")

    if answer is not None:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations, "grounded": grounded}
        )
