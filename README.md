



# ⚡ Enterprise Hybrid RAG with Semantic Caching

A fault-tolerant, enterprise-grade Retrieval-Augmented Generation (RAG) pipeline featuring hybrid search, live telemetry, dynamic model routing, and semantic caching for high-performance AI query resolution.
https://github.com/user-attachments/assets/768964bf-7882-4bc6-bf7b-ada76d443c50
## 🏗️ System Architecture

This pipeline is split into a **FastAPI backend** and a **Streamlit frontend**, integrating multiple advanced retrieval and routing mechanisms:

*   **Hybrid Search (Dense + Sparse):** Combines exact-match lexical retrieval (BM25) with dense vector embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF) to ensure high recall across both keyword-heavy and conceptually complex queries.
*   **Cross-Encoder Re-Ranking:** Surfaces the most contextually relevant hybrid chunks using `ms-marco-MiniLM-L-6-v2`, filtering out low-quality context before LLM generation.
*   **Semantic Caching:** Employs an intelligent ChromaDB caching layer (cosine similarity threshold `0.12`) with a 24-hour Time-To-Live (TTL). Repeated or semantically similar queries bypass the LLM entirely, dropping latency to ~20ms and saving API tokens.
*   **Dynamic Tier Routing & Fallback:** Analyzes query complexity using embedding centroids and lexical heuristics to route prompts to the appropriate model tier (e.g., `gemini-2.5-flash` vs. `gemini-2.0-flash`). Includes automatic fallback resolution to handle `404` (model unavailability) and `429` (quota exhaustion) errors, ultimately defaulting to a Local Extractive Synthesis engine if cloud limits are reached.
*   **RAG Triad Telemetry:** Live, continuous evaluation of generation quality measuring **Context Relevance**, **Groundedness**, and **Answer Relevance** using in-memory vector alignments.

## 🛠️ Tech Stack

*   **Backend:** FastAPI, Uvicorn, Python 3.10+
*   **Frontend:** Streamlit
*   **Vector/Cache Database:** ChromaDB (Persistent)
*   **Embeddings & Re-Ranking:** `sentence-transformers` (Hugging Face)
*   **Sparse Index:** `rank_bm25`
*   **LLM Provider:** Google Gemini API (`google-generativeai`)

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd semantic-router-rag
