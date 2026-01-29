from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DEFAULT_SEPARATORS = ["\n\n", "\n", "▪", "•", "◦", " ", ""]

class TextChunker:
    def __init__(self, chunk_size= 800, chunk_overlap=30):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DEFAULT_SEPARATORS
        )
    def load_docs(self, path: str) -> str:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()[0].page_content
    
    def chunking(self, text: str) -> list[Document]:
        return [
            Document(page_content=chunk) 
            for chunk in self.splitter.split_text(text)
        ]
