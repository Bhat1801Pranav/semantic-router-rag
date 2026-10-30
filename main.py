import os
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI(title="Semantic Cache & Cost-Aware RAG Router")

# Load embedding model & persistent vector store
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

kb_collection = chroma_client.get_or_create_collection(name="knowledge_base")
cache_collection = chroma_client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"}
)

CACHE_DISTANCE_THRESHOLD = 0.12

# --- IN-MEMORY METRICS STORE ---
system_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "model_usage": {
        "cache": 0,
        "gemini-3.5-flash": 0,
        "gemini-3.8-flash": 0
    },
    "latencies_ms": {
        "cache_hit": [],
        "llm_inference": []
    }
}

class QueryRequest(BaseModel):
    query: str

def select_model_tier(query: str) -> tuple[str, str]:
    """Routes query based on complexity heuristics to optimize cost."""
    complex_indicators = [
        "compare", "architecture", "design", "analyze", "tradeoff",
        "why", "how does", "implement", "difference", "evaluate", "pipeline"
    ]
    query_lower = query.lower()
    word_count = len(query.split())

    if word_count > 18 or any(kw in query_lower for kw in complex_indicators):
        return "gemini-3.8-flash", "complex_analytical"
    return "gemini-3.5-flash", "simple_factual"

@app.post("/query")
def process_query(request: QueryRequest):
    start_time = time.time()
    query_text = request.query.strip()
    query_embedding = embedding_model.encode([query_text]).tolist()

    system_metrics["total_queries"] += 1

    # --- STEP 1: CHECK SEMANTIC CACHE ---
    if cache_collection.count() > 0:
        cache_results = cache_collection.query(
            query_embeddings=query_embedding,
            n_results=1
        )
        if cache_results["distances"] and cache_results["distances"][0]:
            distance = cache_results["distances"][0][0]
            if distance <= CACHE_DISTANCE_THRESHOLD:
                cached_meta = cache_results["metadatas"][0][0]
                latency_ms = round((time.time() - start_time) * 1000, 2)

                # Record cache metrics
                system_metrics["cache_hits"] += 1
                system_metrics["model_usage"]["cache"] += 1
                system_metrics["latencies_ms"]["cache_hit"].append(latency_ms)

                return {
                    "query": query_text,
                    "answer": cached_meta["answer"],
                    "cache_hit": True,
                    "matched_cached_query": cache_results["documents"][0][0],
                    "similarity_score": round(1 - distance, 4),
                    "routed_model": "cache (0 tokens)",
                    "latency_ms": latency_ms
                }

    # --- STEP 2: RETRIEVE CONTEXT FROM KNOWLEDGE BASE ---
    system_metrics["cache_misses"] += 1
    kb_results = kb_collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    context_chunks = kb_results["documents"][0] if kb_results["documents"] else []
    context_str = "\n".join(context_chunks)

    # --- STEP 3: COST-AWARE MODEL ROUTING ---
    model_name, complexity_tier = select_model_tier(query_text)

    prompt = (
        f"Answer the user question accurately using the context below.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query_text}"
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        answer_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    # Update routing metrics
    system_metrics["model_usage"][model_name] = system_metrics["model_usage"].get(model_name, 0) + 1

    # --- STEP 4: STORE IN SEMANTIC CACHE ---
    cache_collection.add(
        documents=[query_text],
        embeddings=query_embedding,
        metadatas=[{"answer": answer_text, "model_used": model_name}],
        ids=[str(uuid.uuid4())]
    )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    system_metrics["latencies_ms"]["llm_inference"].append(latency_ms)

    return {
        "query": query_text,
        "answer": answer_text,
        "cache_hit": False,
        "complexity_tier": complexity_tier,
        "routed_model": model_name,
        "retrieved_context": context_chunks,
        "latency_ms": latency_ms
    }

@app.get("/metrics")
def get_evaluation_metrics():
    """Returns real-time pipeline performance and cost-saving metrics."""
    total = system_metrics["total_queries"]
    hits = system_metrics["cache_hits"]
    misses = system_metrics["cache_misses"]

    hit_rate = round((hits / total) * 100, 2) if total > 0 else 0.0

    cache_lats = system_metrics["latencies_ms"]["cache_hit"]
    llm_lats = system_metrics["latencies_ms"]["llm_inference"]

    avg_cache_lat = round(sum(cache_lats) / len(cache_lats), 2) if cache_lats else 0.0
    avg_llm_lat = round(sum(llm_lats) / len(llm_lats), 2) if llm_lats else 0.0

    latency_speedup = round(avg_llm_lat / avg_cache_lat, 1) if (avg_cache_lat > 0 and avg_llm_lat > 0) else 1.0

    return {
        "summary": {
            "total_queries_processed": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": f"{hit_rate}%",
            "estimated_token_cost_saved": f"{hit_rate}%"
        },
        "latency_benchmarks": {
            "avg_cache_latency_ms": avg_cache_lat,
            "avg_llm_latency_ms": avg_llm_lat,
            "cache_speedup_factor": f"{latency_speedup}x faster"
        },
        "model_distribution": system_metrics["model_usage"]
    }