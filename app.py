import streamlit as st
import requests
import json

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Advanced RAG with Semantic Caching",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Enterprise Hybrid RAG with Semantic Cache")
st.caption("Hybrid Search (Dense + BM25), Re-Ranking, Semantic Caching with TTL, and Live Token Streaming")

# --- SIDEBAR METRICS ---
with st.sidebar:
    st.header("📊 Real-Time Telemetry")
    if st.button("Refresh Telemetry", use_container_width=True):
        st.rerun()

    try:
        metrics_resp = requests.get(f"{BACKEND_URL}/metrics", timeout=15)
        if metrics_resp.status_code == 200:
            data = metrics_resp.json()
            summary = data.get("summary", {})
            benchmarks = data.get("latency_benchmarks", {})
            triad = data.get("rag_triad_scores", {})
            dist = data.get("model_distribution", {})

            col1, col2 = st.columns(2)
            col1.metric("Cache Hit Rate", summary.get("cache_hit_rate", "0%"))
            col2.metric("Tokens Saved", summary.get("estimated_token_cost_saved", "0%"))

            st.metric("Cache Speedup Factor", benchmarks.get("cache_speedup_factor", "1.0x"))
            st.metric("Expired Items Evicted (TTL)", summary.get("evicted_expired_items", 0))

            st.divider()
            st.subheader("🎯 RAG Triad Alignment")
            st.progress(float(triad.get("avg_context_relevance", 1.0)), text=f"Context Relevance: {triad.get('avg_context_relevance')}")
            st.progress(float(triad.get("avg_groundedness", 1.0)), text=f"Groundedness: {triad.get('avg_groundedness')}")
            st.progress(float(triad.get("avg_answer_relevance", 1.0)), text=f"Answer Relevance: {triad.get('avg_answer_relevance')}")

            st.divider()
            st.write("**Model Routing Distribution**")
            for model_name, count in dist.items():
                st.write(f"- `{model_name}`: **{count}** calls")
    except Exception as e:
        st.error(f"Backend offline: {e}")

# --- QUERY INTERFACE ---
st.subheader("Ask the Knowledge Base")
user_query = st.text_input(
    "Query:",
    placeholder="e.g., What is Project Alpha? or Compare architecture tradeoffs..."
)

if st.button("Run Query", type="primary", use_container_width=True) and user_query.strip():
    status_box = st.container()
    answer_box = st.empty()
    trace_box = st.container()

    try:
        response = requests.post(
            f"{BACKEND_URL}/query-stream",
            json={"query": user_query.strip()},
            stream=True,
            timeout=60
        )

        full_answer = ""
        meta_data = {}
        end_stats = {}

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                payload = raw_line[6:].strip()
                if payload == "[DONE]":
                    break

                chunk = json.loads(payload)

                if chunk.get("type") == "metadata":
                    meta_data = chunk
                    with status_box:
                        st.write("---")
                        c1, c2, c3 = st.columns(3)
                        if meta_data.get("cache_hit"):
                            c1.success("⚡ Cache Hit (0 Tokens)")
                        else:
                            c1.info("🔍 Cache Miss (Hybrid Retrieval + LLM)")
                        c2.metric("Routed Model", meta_data.get("routed_model", "Unknown"))
                        latency_val = f"{meta_data.get('latency_ms')} ms" if meta_data.get("cache_hit") else "Streaming..."
                        c3.metric("Latency", latency_val)

                elif chunk.get("type") == "token":
                    full_answer += chunk.get("content", "")
                    answer_box.markdown(full_answer + "▌")

                elif chunk.get("type") == "end_stats":
                    end_stats = chunk

                elif chunk.get("type") == "error":
                    st.error(f"LLM Error: {chunk.get('detail')}")

        answer_box.markdown(full_answer)

        with trace_box:
            with st.expander("🔍 Pipeline Trace & RAG Triad Details", expanded=True):
                if meta_data.get("cache_hit"):
                    st.write(f"**Matched Query:** `{meta_data.get('matched_cached_query')}`")
                    st.write(f"**Similarity Score:** `{meta_data.get('similarity_score')}`")
                else:
                    st.write(f"**Complexity Tier:** `{meta_data.get('complexity_tier')}`")
                    st.write("**Top Re-Ranked Hybrid Chunks:**")
                    for idx, c in enumerate(meta_data.get("retrieved_context", []), 1):
                        st.info(f"Chunk {idx}: {c}")

                if end_stats:
                    st.write(f"**Total Generation Latency:** `{end_stats.get('latency_ms')} ms`")
                    st.json(end_stats.get("rag_triad", {}))

    except Exception as err:
        st.error(f"Connection failed: {err}")