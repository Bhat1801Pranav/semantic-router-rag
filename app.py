import streamlit as st
import requests
import json

BACKEND_URL = "https://semantic-router-rag.onrender.com"

st.set_page_config(page_title="Hybrid RAG Search", layout="centered")
st.title("Enterprise Hybrid RAG Pipeline")
st.write("Powered by FastAPI, ChromaDB, and Gemini AI")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Searching and generating response..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query-stream",
                    json={"query": query},
                    stream=True,
                    timeout=120
                )
                
                if response.status_code == 200:
                    st.markdown("### Answer")
                    answer_placeholder = st.empty()
                    full_response = ""
                    
                    # Process the Server-Sent Events (SSE) stream line by line
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            data_str = line[6:] # Remove the "data: " prefix
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                # Parse the JSON payload
                                data_json = json.loads(data_str)
                                
                                # Extract and append only the text tokens
                                if data_json.get("type") == "token":
                                    full_response += data_json.get("content", "")
                                    # Update the UI with the clean text
                                    answer_placeholder.markdown(full_response)
                                    
                            except json.JSONDecodeError:
                                continue
                                
                    if not full_response.strip():
                        st.warning("The backend returned an empty response. Check if ChromaDB is populated.")
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
    else:
        st.warning("Please enter a query first.")