import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Target the persistent chroma_db folder used by your backend
CHROMA_PATH = "./chroma_db"
os.makedirs(CHROMA_PATH, exist_ok=True)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use default all-MiniLM-L6-v2 embeddings matching sentence-transformers
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# 2. Get or recreate the collection
collection_name = "documents"
try:
    client.delete_collection(name=collection_name)
    print(f"Cleared existing '{collection_name}' collection.")
except Exception:
    pass

collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

# 3. Define the knowledge base documents
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

# 4. Ingest and persist
print(f"Indexing {len(documents)} documents into ChromaDB...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Ingestion complete! Total items in collection:", collection.count())