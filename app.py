import streamlit as st
import requests

# Your live Render backend URL
BACKEND_URL = "https://semantic-router-rag.onrender.com"

st.set_page_config(page_title="Hybrid RAG Search", layout="centered")
st.title("Enterprise Hybrid RAG Pipeline")
st.write("Powered by FastAPI, ChromaDB, and Gemini AI")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Connecting to backend and searching..."):
            try:
                # Pointing directly to your custom /query-stream endpoint
                response = requests.post(f"{BACKEND_URL}/query-stream", json={"query": query})
                
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Answer")
                    
                    # Checks for common response keys, otherwise displays the raw output
                    if "answer" in data:
                        st.write(data["answer"])
                    elif "response" in data:
                        st.write(data["response"])
                    else:
                        st.write(data) 
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend. Error: {e}")
    else:
        st.warning("Please enter a query first.")