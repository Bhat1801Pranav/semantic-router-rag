import os
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize persistent ChromaDB vector store on disk
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

def ingest_documents():
    documents = [
        "Project Alpha is an enterprise AI framework developed to optimize cloud infrastructure and reduce latency.",
        "Semantic caching stores previous LLM queries and their vector embeddings to serve identical or similar requests instantly without hitting the API.",
        "Cost-aware routing dynamically evaluates incoming query complexity and routes simpler prompts to smaller, cheaper models.",
        "Labmentix focuses on building advanced, production-grade AI architectures and robust engineering pipelines."
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    embeddings = embedding_model.encode(documents).tolist()
    
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Successfully ingested {len(documents)} documents into persistent ChromaDB!")

if __name__ == "__main__":
    ingest_documents()