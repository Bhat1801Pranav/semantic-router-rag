import chromadb
from sentence_transformers import SentenceTransformer

print("Connecting to ChromaDB...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

kb_collection = chroma_client.get_or_create_collection(name="knowledge_base")
cache_collection = chroma_client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"}
)

# Clear old cached empty responses
if cache_collection.count() > 0:
    cache_collection.delete(ids=cache_collection.get()["ids"])
    print("Cleared stale entries from semantic_cache.")

documents = [
    "Project Alpha is the next-generation autonomous edge AI platform designed for low-latency inferencing on decentralized micro-servers. It is targeted for launch in Q3 2027 and is led by the core Systems & Applied AI team.",
    "Project Alpha architecture utilizes a hybrid retrieval-augmented generation (RAG) system paired with semantic routing to minimize API token consumption and improve query response times under 200 milliseconds.",
    "Project Beta is a data synchronization pipeline that facilitates automated ETL transfers between local SQLite instances and distributed analytical warehouses.",
    "The Enterprise RAG Pipeline uses ChromaDB as its vector store, BM25Okapi for sparse keyword retrieval, CrossEncoder (ms-marco-MiniLM-L-6-v2) for reranking, and sentence-transformers (all-MiniLM-L6-v2) for local dense embeddings."
]

metadatas = [
    {"source": "project_alpha_spec.pdf", "topic": "Project Alpha Overview"},
    {"source": "project_alpha_architecture.md", "topic": "Project Alpha Tech Stack"},
    {"source": "project_beta_overview.txt", "topic": "Project Beta Overview"},
    {"source": "rag_infra_doc.md", "topic": "Infrastructure Overview"}
]

ids = [f"doc_{i+1}" for i in range(len(documents))]
embeddings = embedding_model.encode(documents).tolist()

kb_collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Ingestion complete! Total items in knowledge_base:", kb_collection.count())