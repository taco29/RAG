import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

client = QdrantClient("http://localhost:6333")

def create_vectorstore(documents, embedding):
    ids = [str(uuid.uuid4()) for _ in documents]
    try:
        vector_size = embedding.embedding_dimension
    except Exception:
        try:
            sample = embedding.embed_query("test")
        except Exception:
            sample = embedding.embed_documents(["test"])[0]
        vector_size = len(sample)
    client.recreate_collection(
        collection_name="documents",
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name="documents",
        url="http://localhost:6333",
        ids=ids
    )
    return vector_store