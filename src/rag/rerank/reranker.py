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


    @staticmethod
    def _clone_docs(docs: List[dict]) -> List[dict]:
        cloned = []
        for doc in docs or []:
            if hasattr(doc, "model_dump"):
                cloned.append(doc.model_dump())
            elif isinstance(doc, dict):
                cloned.append(dict(doc))
        return cloned


    def rank(self, query: str, docs: List[dict]) -> List[dict]:
        return self.reranker.run(query, self._clone_docs(docs))


    def run(self,query: str, docs: List[dict],score=settings.reranker_min_score,top_k:int=settings.reranker_top_k):
        ranked_docs = self.rank(query, docs)
        if score is not None:
            ranked_docs = [doc for doc in ranked_docs if (doc.get("rerank_score") or 0.0) >= score]
        if top_k is not None and int(top_k) > 0:
            ranked_docs = ranked_docs[:top_k]
        return ranked_docs



if __name__ == "__main__":

    a = [1,2]
