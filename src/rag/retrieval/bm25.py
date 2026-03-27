import jieba
from rank_bm25 import BM25Okapi

from core.settings import settings
from src.database.es import ElasticsearchClient
from utils.utils import is_chinese






class BaseBM25Retrieval:
    def run(self,search_queries: list[str], top_k=settings.retriever_top_k, filters=None):
        raise NotImplementedError


class BM25LiteRetriever(BaseBM25Retrieval):

    def __init__(self, documents):
        if not documents:
            raise ValueError("documents不能为空")
        self.docs = documents
        self.corpus = []
        for  doc in documents:
            if is_chinese(doc["content"]):
               self.corpus.append(jieba.lcut(doc["content"]))
            else:
                self.corpus.append(doc["content"].lower().split())
        self.bm25 = BM25Okapi(self.corpus)




    def run(self, search_queries: list[str], top_k=settings.retriever_top_k,filters=None):
        doc_scores = {}
        for query in search_queries:
            if not query or not query.split():
                continue
            if is_chinese(query):
                tokenized_query = jieba.lcut(query)
            else:
                tokenized_query = query.lower().split()

            scores = self.bm25.get_scores(tokenized_query)
            for doc, score in zip(self.docs, scores):

                # metadata过滤
                if filters:
                    skip = False
                    for k, v in filters.items():
                        if doc["metadata"].get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue

                doc_id = doc["content"]

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "bm25_score": float(score),
                        "node_id":doc['node_id']
                    }
                else:
                    doc_scores[doc_id]["bm25_score"] = max(
                        doc_scores[doc_id]["bm25_score"],
                        float(score)
                    )

        results = list(doc_scores.values())
        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]




class ESRetriever(BaseBM25Retrieval):

    def __init__(self, es_client:ElasticsearchClient):
        self.es_client = es_client

    def run(self, search_queries: list[str], top_k=settings.retriever_top_k, filters=None):

        should = [{"match": {"content": query}} for query in search_queries if query and query.split() ]

        filter_clause = []

        if filters:
            for k, v in filters.items():
                filter_clause.append({
                    "term": {f"metadata.{k}.keyword": v}
                })

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    "filter": filter_clause
                }
            }
        }
        res = self.es_client.es.search(index=self.es_client.index, body=body)

        results = []
        for hit in res["hits"]["hits"]:
            results.append({
                "content": hit["_source"]["content"],
                "metadata": hit["_source"]['metadata'],
                "bm25_score": hit["_score"],
                "node_id": hit["_score"]['node_id']
            })

        return results


