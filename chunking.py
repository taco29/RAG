from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_text(path):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return docs[0].page_content

def chunking(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "▪",
            "•",
            "◦",
            " ",
            ""
        ]
    )
    chunks = text_splitter.split_text(documents)
    return chunks

def create_documents(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs