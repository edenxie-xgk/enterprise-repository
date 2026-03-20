from typing import List

from langchain_core.language_models import BaseChatModel

from core.settings import settings
from src.models.reranker import reranker_model
from src.rag.rerank.corss_encoder_rerander import CrossEncoderReRanker
from src.rag.rerank.llm_reranker import LLMReRanker


class Reranker:
    def __init__(self, llm:BaseChatModel = None,top_k: int= settings.reranker_top_k):
        if  settings.reranker_type == "llm":
            self.reranker = LLMReRanker(llm=llm, top_k=top_k)
        elif settings.reranker_type == "cross-encoder":
            self.reranker = CrossEncoderReRanker(model=reranker_model, top_k=top_k)
        else:
            raise Exception("reranker type error")


    def run(self,query: str, docs: List[dict],score=settings.reranker_min_score):
        return [doc for doc in self.reranker.run(query, docs) if doc["rerank_score"] >= score]



if __name__ == "__main__":

    a = [1,2]
