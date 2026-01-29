from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nomalize: bool = True
    ):
        self.model_name = model_name
        self.nomalize = nomalize
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.model_name,
            encode_kwargs={"normalize_embeddings": self.nomalize}
        )
    def get(self):
        return self.embedding
    