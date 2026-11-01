import streamlit as st
import requests

st.title("Enterprise Hybrid RAG Pipeline 💬")
st.caption("Powered by FastAPI, ChromaDB, and Gemini AI")

BACKEND_URL = "https://semantic-router-rag.onrender.com/query"

# 1. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render existing conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new user input
if prompt := st.chat_input("Ask a question about the documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with requests.post(
                BACKEND_URL,
                json={"query": prompt},
                stream=True,
                timeout=120
            ) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")