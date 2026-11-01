from sentence_transformers import SentenceTransformer
import main

print("Connecting directly to main.py's kb_collection and cache_collection...")

# 1. Clear old cached "No relevant documents found" responses
try:
    cached_items = main.cache_collection.get()
    if cached_items and cached_items.get("ids"):
        main.cache_collection.delete(ids=cached_items["ids"])
        print(f"Cleared {len(cached_items['ids'])} stale entries from cache_collection.")
except Exception as e:
    print(f"Cache clear note: {e}")

# 2. Define the knowledge base documents
documents = [
    (
        "Project Alpha is the next-generation autonomous edge AI platform designed for low-latency inferencing "
        "on decentralized micro-servers. It is targeted for launch in Q3 2027 and is led by the core Systems & Applied AI team."
    ),
    (
        "Project Alpha architecture utilizes a hybrid retrieval-augmented generation (RAG) system "
        "paired with semantic routing to minimize API token consumption and improve query response times under 200 milliseconds."
    ),
    (
        "Project Beta is a data synchronization pipeline that facilitates automated ETL transfers between local SQLite instances "
        "and distributed analytical warehouses."
    ),
    (
        "The Enterprise RAG Pipeline uses ChromaDB as its vector store and sentence-transformers for local dense embeddings, "
        "enabling fully offline vector indexing."
    )
]

metadatas = [
    {"source": "project_alpha_spec.pdf", "topic": "Project Alpha Overview"},
    {"source": "project_alpha_architecture.md", "topic": "Project Alpha Tech Stack"},
    {"source": "project_beta_overview.txt", "topic": "Project Beta Overview"},
    {"source": "rag_infra_doc.md", "topic": "Infrastructure Overview"}
]

ids = [f"doc_{i+1}" for i in range(len(documents))]

# 3. Compute embeddings using the exact model from main.py (or all-MiniLM-L6-v2)
embedder = getattr(main, "embedder", None) or getattr(main, "embedding_model", None) or SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents).tolist()

# 4. Upsert directly into main.kb_collection
main.kb_collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Ingestion complete! Total items in kb_collection:", main.kb_collection.count())