from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EmbeddingManager:
    def __init__(self, embedded_model="BAAI/bge-small-en"):
        self.embedded_model = embedded_model
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedded_model)

    def get_embed_model(self):
        return self.embed_model


    def get_embeddings(self, query_str):
        query_embedding = self.embed_model.get_query_embedding(query_str)
        return query_embedding



