# ⚡ Cost-Aware RAG Pipeline with Semantic Caching

A production-grade Retrieval-Augmented Generation (RAG) architecture built with **FastAPI**, **ChromaDB**, and **Google Gemini**. The system achieves sub-30ms response latencies and significant API token cost savings by implementing high-dimensional vector caching and dynamic, complexity-based model routing.

---

## 🌟 Key Features

* **High-Throughput Semantic Caching**: Evaluates queries against cached vector embeddings using cosine distance thresholding ($\le 0.12$ / $\ge 0.88$ similarity) via `all-MiniLM-L6-v2`. Semantic matches bypass the LLM entirely, serving grounded answers with **~20ms latency at 0 token cost**.
* **Cost-Aware Heuristic Router**: Automatically routes cache-miss queries based on lexical complexity, prompt length, and analytical keywords to the optimal model tier:
  * **Simple / Factual**: Fast model tier (`gemini-3.5-flash`).
  * **Complex / Analytical**: Deep reasoning tier (`gemini-3.8-flash`).
* **Live Pipeline Evaluation Metrics**: In-memory telemetry exposed via a dedicated REST endpoint tracking real-time cache hit ratios, estimated token savings, latency comparisons, and model utilization distributions.
* **Interactive Frontend Dashboard**: Built with **Streamlit** for real-time query tracing, similarity inspection, and live metric tracking.

---

## 🏗️ System Architecture