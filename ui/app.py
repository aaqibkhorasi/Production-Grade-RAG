import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ask My Docs - SBA")
st.title("Ask My Docs: SBA Loan Requirements")

question = st.text_input("Ask a question about SBA 7(a)/504 loan requirements")

if st.button("Ask") and question:
    with st.spinner("Retrieving answer..."):
        response = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)

    if response.status_code == 200:
        data = response.json()
        st.markdown("### Answer")
        st.write(data["answer"])
        st.markdown("### Sources")
        for citation in data["citations"]:
            label = f"{citation['source_doc']} (effective {citation['effective_date']})"
            with st.expander(label):
                st.write(citation["chunk_text"])
    else:
        st.error(f"Request failed: {response.status_code} {response.text}")
