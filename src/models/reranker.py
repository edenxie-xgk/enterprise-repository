from sentence_transformers import CrossEncoder

from core.settings import settings

reranker_model = CrossEncoder(settings.reranker_model)

