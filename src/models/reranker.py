from sentence_transformers import CrossEncoder

from core.settings import settings


_reranker_model = None


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.reranker_model)
    return _reranker_model


