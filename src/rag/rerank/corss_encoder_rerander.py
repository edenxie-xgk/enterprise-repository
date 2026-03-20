from core.settings import settings
from src.models.reranker import reranker_model
class  CrossEncoderReRanker:
    def __init__(self, model = reranker_model, top_k = settings.reranker_top_k):
        self.model = model
        self.top_k = top_k


    def run(self, query: str, docs: list[dict])->list[dict]:
        """重排文档"""
        # 构造输入对

        pairs = [
            (query, doc["content"][:settings.reranker_max_len])  # 截断（非常重要）
            for doc in docs
        ]

        # 批量打分（关键优化点）
        scores = self.model.predict(pairs)

        # 写回score
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        # 排序
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return docs[:self.top_k]

