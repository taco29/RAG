from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DEFAULT_SEPARATORS = ["\n\n", "\n", "▪", "•", "◦", " ", ""]

class TextChunker:
    def __init__(self, chunk_size, chunk_overleap):
        self.chunk_size = chunk_size
        self.chunk_overleap = chunk_overleap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size= self.chunk_size,
            chunk_overleap=self.chunk_overleap,
            separators= DEFAULT_SEPARATORS                                                          
        )
    def load_and_chunk(self, path: str):
        loader = TextLoader(path, encoding="utf-8")
        text = loader.load()[0].page_content
        chunks = self.splitter.split_text(text)
        return [Document(page_content=c) for c in chunks]


class create_documents:
    def __init__(self, chunks):
        self.chunks= chunks

    def create_docs(self):
        docs = Document(page_content=self.chunks)