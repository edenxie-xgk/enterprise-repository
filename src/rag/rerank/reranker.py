from typing import List

from langchain_core.language_models import BaseChatModel

from core.settings import settings
from src.rag.rerank.corss_encoder_rerander import CrossEncoderReRanker
from src.rag.rerank.llm_reranker import LLMReRanker


class Reranker:
    def __init__(self, llm:BaseChatModel = None):
        if  settings.reranker_type == "llm":
            self.reranker = LLMReRanker(llm=llm)
        elif settings.reranker_type == "cross-encoder":
            self.reranker = CrossEncoderReRanker()
        else:
            raise Exception("reranker type error")


    def run(self,query: str, docs: List[dict],score=settings.reranker_min_score,top_k:int=settings.reranker_top_k):
        return [doc for doc in self.reranker.run(query, docs) if doc["rerank_score"] >= score][:top_k]



if __name__ == "__main__":

    a = [1,2]
