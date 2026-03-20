from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.settings import settings

embed_model = HuggingFaceEmbedding(
    model_name=settings.embedding_model
)

