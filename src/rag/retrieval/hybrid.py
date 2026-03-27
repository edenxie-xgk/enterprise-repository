from core.settings import settings
from src.rag.retrieval.bm25 import BaseBM25Retrieval
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.rrf import rrf_fusion


class HybridRetriever:

    def __init__(self, dense_retriever:DenseRetriever, bm25_retriever:BaseBM25Retrieval):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def run(self, queries):
        dense_results = self.dense.run(queries)
        bm25_results = self.bm25.run(queries)
        fused = rrf_fusion([dense_results, bm25_results])
        return fused