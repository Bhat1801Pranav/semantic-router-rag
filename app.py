import streamlit as st
import requests

st.title("Enterprise Hybrid RAG Pipeline 💬")
st.caption("Powered by FastAPI, ChromaDB, and Gemini AI")

# 1. Initialize the chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display previous chat messages when the app reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Accept user input at the bottom of the screen
if prompt := st.chat_input("Ask a question about the documents..."):
    
    # Display the user's question instantly
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save the user's question to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Fetch and display the AI's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # ---> PASTE YOUR EXISTING BACKEND REQUEST HERE <---
            # Example: response = requests.post("YOUR_RENDER_URL", json={"query": prompt})
            # answer = response.text
            
            # (Replace this placeholder variable with your actual backend answer)
            answer = "This is where your FastAPI backend answer will appear."
            
            message_placeholder.markdown(answer)
            
            # Save the AI's answer to history so it stays on screen
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Backend Error: {e}")