"""
vector_store.py
─────────────────────────────────────────────────────────────────
1. Chunks extracted page text using LangChain's RecursiveCharacterTextSplitter.
2. Embeds chunks using ChromaDB's built-in embedding (sentence-transformers).
3. Stores and retrieves chunks per doc_id so every uploaded report
   gets its own isolated collection in ChromaDB.
"""

import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from extractor import extract_full_text

load_dotenv()

CHROMA_DIR   = os.getenv("CHROMA_DIR", "chroma_db")
CHUNK_SIZE   = 800     # characters per chunk
CHUNK_OVERLAP = 120    # overlap between chunks to preserve context


# ── Singleton ChromaDB client ──────────────────────────────────────────────────
_client: chromadb.PersistentClient | None = None

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


# ── Chunk text ─────────────────────────────────────────────────────────────────
def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Split page texts into smaller overlapping chunks.

    Input:  [{ "page": 1, "text": "..." }, ...]
    Output: [{ "page": 1, "chunk_id": "1_0", "text": "..." }, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "page":     page["page"],
                "chunk_id": f"{page['page']}_{i}",
                "text":     split.strip(),
            })

    return chunks




# ── Embed and store ────────────────────────────────────────────────────────────
def store_chunks(doc_id: str, chunks: List[Dict]) -> int:
    """
    Embed chunks and store them in a ChromaDB collection named after doc_id.
    Returns the number of chunks stored.
    """
    client = get_client()

    # Each doc gets its own collection — delete old one if re-uploading
    try:
        client.delete_collection(name=doc_id)
    except Exception:
        pass  # collection didn't exist yet

    collection = client.create_collection(
        name=doc_id,
        embedding_function=DefaultEmbeddingFunction(),   # uses all-MiniLM-L6-v2
        metadata={"hnsw:space": "cosine"},
    )
    

    # ChromaDB add() in batches (max 5000 per call)
    BATCH = 500
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        collection.add(
            ids        = [c["chunk_id"]               for c in batch],
            documents  = [c["text"]                   for c in batch],
            metadatas  = [{"page": c["page"]}         for c in batch],
        )
        
        '''
        This is for checking embeddings
        
    data = collection.get(
    include=["embeddings", "documents", "metadatas"]
    )

    print(data["embeddings"])

        '''
    return len(chunks)






# ── Retrieve ───────────────────────────────────────────────────────────────────
def retrieve(doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top_k most relevant chunks for a query from a doc's collection.

    Returns:
        [{ "text": "...", "page": 3, "score": 0.87 }, ...]
    """
    client = get_client()

    try:
        collection = client.get_collection(
            name=doc_id,
            embedding_function=DefaultEmbeddingFunction(),
        )
    except Exception:
        raise ValueError(f"No vector store found for doc_id: {doc_id}")

    results = collection.query(
        query_texts = [query],
        n_results   = min(top_k, collection.count()),
        include     = ["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":    doc,
            "page":    meta.get("page", 0),
            "score":   round(1 - dist, 3),  # cosine distance → similarity
        })

    return chunks


data=extract_full_text('./pdf/T1.2.pdf')

chunks=chunk_pages(data)
lengthing=store_chunks("91101",chunks)
print(lengthing)

# chunksstor=retrieve("91101","End the action",6)
# ans = '\n'.join(str(item) for item in chunksstor)
# print(ans)




# ── Delete ─────────────────────────────────────────────────────────────────────
def delete_doc(doc_id: str):
    """Remove a document's vector store collection."""
    try:
        get_client().delete_collection(name=doc_id)
    except Exception:
        pass
