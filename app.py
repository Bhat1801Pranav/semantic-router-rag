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
        with st.spinner("Searching and re-ranking..."):
            try:
                # Adjust the "/query" endpoint below if your main.py uses a different route name (like "/search" or "/chat")
                response = requests.post(f"{BACKEND_URL}/query", json={"query": query})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Assuming your backend returns {"answer": "...", "sources": [...] }
                    # Adjust these keys if your main.py returns different dictionary keys
                    if "answer" in data:
                        st.markdown("### Answer")
                        st.write(data["answer"])
                    else:
                        st.write(data)
                        
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend. Make sure the Render service is awake! Error: {e}")
    else:
        st.warning("Please enter a query first.")