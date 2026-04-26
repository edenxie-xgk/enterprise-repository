from sentence_transformers import CrossEncoder

from core.settings import settings

from langchain_greennode import GreenNodeRerank



reranker_model = CrossEncoder(settings.reranker_model)


