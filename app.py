import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📄")
st.title("📄 RAG Knowledge Assistant")
st.markdown("Ask questions about your documents.")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question:
        with st.spinner("Searching and generating answer..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                if response.status_code == 200:
                    data = response.json()
                    st.write("**Answer:**", data["answer"])
                    with st.expander("See sources"):
                        for source in data["sources"]:
                            st.write(f"- {source}")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter a question.")