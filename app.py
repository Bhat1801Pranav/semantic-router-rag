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
                # Use stream=True to support FastAPI StreamingResponse
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
                    
                    # Read chunks as they arrive from the stream
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            answer_placeholder.markdown(full_response)
                    
                    # If the backend returned a raw JSON string, format it cleanly
                    if full_response.strip().startswith("{") and full_response.strip().endswith("}"):
                        try:
                            parsed = json.loads(full_response)
                            if "answer" in parsed:
                                answer_placeholder.markdown(parsed["answer"])
                            elif "response" in parsed:
                                answer_placeholder.markdown(parsed["response"])
                        except json.JSONDecodeError:
                            pass
                            
                    if not full_response.strip():
                        st.warning("The backend returned an empty response. Check your Render logs to ensure GEMINI_API_KEY is set and ChromaDB is populated.")
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
    else:
        st.warning("Please enter a query first.")