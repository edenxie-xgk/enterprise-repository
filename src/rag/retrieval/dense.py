from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever

from core.settings import settings


class DenseRetriever:

    def __init__(self, vector_store,storage_context, top_k: int = settings.retriever_top_k):
        self.vector_store = vector_store
        self.top_k = top_k
        self.storage_context = storage_context
        self.index = VectorStoreIndex.from_vector_store(
            storage_context=self.storage_context,
            vector_store=self.vector_store,
        )
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
            filters=None,
        )

    @staticmethod
    def _value_matches(actual, expected) -> bool:
        if isinstance(expected, (list, tuple, set)):
            return any(DenseRetriever._value_matches(actual, item) for item in expected)
        if actual == expected:
            return True
        return str(actual) == str(expected)

    @staticmethod
    def _matches_filters(metadata: dict, filters=None) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            actual = metadata.get(key)
            if not DenseRetriever._value_matches(actual, expected):
                return False
        return True

    def run(self,search_queries:list[str],filters=None,top_k:int=None,score=settings.retrieval_min_score):
        all_results = []
        if top_k:
            self.retriever.similarity_top_k = top_k
        for query in search_queries:
            if not query:
                continue
            self.retriever._filters = filters
            results = self.retriever.retrieve(query)
            for node in results:
                if node.score < score:
                    continue
                # metadata过滤
                if not self._matches_filters(node["metadata"], filters):
                    continue

                doc = {
                    "content": node.text,
                    "metadata": node.metadata,
                    "dense_score": node.score,
                    "node_id": node.node_id
                }
                all_results.append(doc)

        return all_results


