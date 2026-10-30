import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Cost-Aware RAG with Semantic Caching",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Semantic Cache & Cost-Aware RAG Pipeline")
st.caption("Sub-30ms vector caching with ChromaDB and dynamic model routing via Gemini")

# Sidebar for Real-Time Metrics
with st.sidebar:
    st.header("📊 Pipeline Performance")
    if st.button("Refresh Metrics", use_container_width=True):
        st.rerun()

    try:
        metrics_resp = requests.get(f"{BACKEND_URL}/metrics", timeout=5)
        if metrics_resp.status_code == 200:
            metrics = metrics_resp.json()
            summary = metrics.get("summary", {})
            benchmarks = metrics.get("latency_benchmarks", {})
            dist = metrics.get("model_distribution", {})

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Cache Hit Rate", summary.get("cache_hit_rate", "0%"))
            col_m2.metric("Tokens Saved", summary.get("estimated_token_cost_saved", "0%"))

            st.metric("Speedup Factor", benchmarks.get("cache_speedup_factor", "1.0x"))
            
            st.divider()
            st.write("**Latency Breakdown**")
            st.write(f"- Avg Cache Latency: `{benchmarks.get('avg_cache_latency_ms', 0)} ms`")
            st.write(f"- Avg LLM Latency: `{benchmarks.get('avg_llm_latency_ms', 0)} ms`")

            st.divider()
            st.write("**Model Routing Distribution**")
            for model_name, count in dist.items():
                st.write(f"- `{model_name}`: **{count}** calls")
        else:
            st.warning("Could not fetch metrics from backend.")
    except Exception as e:
        st.error(f"Backend offline: {e}")

# Query Interface
st.subheader("Query the Knowledge Base")
user_query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is Project Alpha? or Compare architecture tradeoffs..."
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    submit = st.button("Run Query", type="primary", use_container_width=True)

if submit and user_query.strip():
    with st.spinner("Processing query through RAG pipeline..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": user_query.strip()},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                
                # Top Status Indicators
                st.write("---")
                c1, c2, c3 = st.columns(3)
                
                if data.get("cache_hit"):
                    c1.success("⚡ Cache Hit (0 Tokens)")
                else:
                    c1.info("🔍 Cache Miss (LLM Generated)")

                c2.metric("Routed Model / Tier", data.get("routed_model", "Unknown"))
                c3.metric("Latency", f"{data.get('latency_ms', 0)} ms")

                # Main Answer Display
                st.markdown("### Answer")
                st.write(data.get("answer", ""))

                # Debug / Trace Details
                with st.expander("Inspect Retrieval & Pipeline Trace"):
                    if data.get("cache_hit"):
                        st.write(f"**Matched Query:** `{data.get('matched_cached_query')}`")
                        st.write(f"**Similarity Score:** `{data.get('similarity_score')}`")
                    else:
                        st.write(f"**Complexity Tier:** `{data.get('complexity_tier')}`")
                        st.write("**Retrieved Knowledge Base Context:**")
                        for idx, chunk in enumerate(data.get("retrieved_context", []), 1):
                            st.info(f"Chunk {idx}: {chunk}")
            else:
                st.error(f"Server Error ({response.status_code}): {response.text}")

        except Exception as err:
            st.error(f"Failed to connect to FastAPI backend: {err}")