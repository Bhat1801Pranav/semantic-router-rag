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
                # Smart auto-detect: tries the most common FastAPI endpoint names
                endpoints = ["/search", "/query", "/chat", "/ask", "/generate", "/api/search"]
                success = False
                
                for ep in endpoints:
                    # Attempt 1: Assumes backend expects {"query": "..."}
                    response = requests.post(f"{BACKEND_URL}{ep}", json={"query": query})
                    
                    if response.status_code == 200:
                        data = response.json()
                        success = True
                        
                        st.markdown("### Answer")
                        if "answer" in data:
                            st.write(data["answer"])
                        elif "response" in data:
                            st.write(data["response"])
                        elif "result" in data:
                            st.write(data["result"])
                        else:
                            st.write(data) # Fallback to display raw JSON
                        break
                    
                    # Attempt 2: If 422 Unprocessable Entity, backend endpoint exists but expects {"text": "..."} instead
                    elif response.status_code == 422:
                        response_alt = requests.post(f"{BACKEND_URL}{ep}", json={"text": query})
                        if response_alt.status_code == 200:
                            data = response_alt.json()
                            success = True
                            
                            st.markdown("### Answer")
                            if "answer" in data: st.write(data["answer"])
                            elif "response" in data: st.write(data["response"])
                            else: st.write(data)
                            break
                
                if not success:
                    st.error("Backend Error 404: Could not find the correct endpoint. Open your `main.py` file in VS Code, look for the `@app.post(...)` line, and verify the route name!")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend. Error: {e}")
    else:
        st.warning("Please enter a query first.")