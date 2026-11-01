import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load your local .env file to grab GEMINI_API_KEY
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found! Make sure it is set in your .env file.")

genai.configure(api_key=api_key)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Delete old 384-dimension collections so ChromaDB resets cleanly to 3072 dimensions
for col_name in ["knowledge_base", "semantic_cache"]:
    try:
        chroma_client.delete_collection(name=col_name)
        print(f"Deleted old '{col_name}' collection.")
    except Exception:
        pass

kb_collection = chroma_client.get_or_create_collection(name="knowledge_base")
cache_collection = chroma_client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"}
)

# 3. Read and chunk sample.txt alongside the original demo docs
documents = [
    "Project Alpha is the next-generation autonomous edge AI platform designed for low-latency inference.",
    "Project Beta is a data synchronization pipeline that facilitates automated ETL transfers between local SQLite instances and distributed analytical warehouses.",
    "The Enterprise RAG Pipeline uses ChromaDB as its vector store, BM25Okapi for sparse keyword retrieval, and Gemini AI for embeddings and generation."
]
metadatas = [
    {"source": "project_alpha_spec.pdf", "topic": "Project Alpha Overview"},
    {"source": "project_beta_overview.txt", "topic": "Project Beta Overview"},
    {"source": "rag_infra_doc.md", "topic": "Infrastructure Overview"}
]

if os.path.exists("sample.txt"):
    with open("sample.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Split by paragraphs or group lines into meaningful chunks
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    chunk_size = 5  # Group every 5 non-empty lines into a chunk
    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i:i + chunk_size])
        documents.append(chunk)
        metadatas.append({"source": "sample.txt", "topic": f"Knowledge Chunk {i//chunk_size + 1}"})

print(f"Embedding {len(documents)} chunks with models/gemini-embedding-001 (3072-dim)...")

# 4. Generate 3072-dim embeddings in batches via Gemini API
embeddings = []
batch_size = 50
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=batch
    )
    embeddings.extend(response["embedding"])

ids = [f"doc_{i+1}" for i in range(len(documents))]

kb_collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Ingestion complete! Total items in knowledge_base:", kb_collection.count())