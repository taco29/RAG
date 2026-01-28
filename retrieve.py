from chunking import chunking, load_text , create_documents
from embedding import get_embedding
from qdrant import create_vectorstore


def retriever(path, k):
    documents = load_text(path)
    chunks = chunking(documents, chunk_size=500, chunk_overlap=50)
    docs = create_documents(chunks)

    embedding = get_embedding()
    vector_store = create_vectorstore(docs, embedding)

    retriever_instance = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever_instance

def get_chunks(path):
    documents = load_text(path)
    chunks = chunking(documents, chunk_size=500, chunk_overlap=50)
    docs = create_documents(chunks)
    return docs