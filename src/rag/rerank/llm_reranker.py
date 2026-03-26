import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from core.settings import settings
from src.prompts.rag.rerank_prompt import RERANK_PROMPT


class LLMReRanker:
    def __init__(self, llm:BaseChatModel):
        self.llm = llm


    def score_doc(self, query: str, doc: dict):
        """获取文档的相关性分数"""
        prompt = RERANK_PROMPT.format(
            query=query,
            document=doc['content'][:settings.reranker_max_len] # 限制长度（很重要）
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            data = json.loads(content)

            return float(data.get("score", 0.0))

        except Exception:
            return 0.0

    def run(self, query: str, docs: list[dict])->list[dict]:
        """获取每个文档对应的相关性分数并进行排序"""
        scored_docs = []
        for doc in docs:
            score = self.score_doc(query, doc)
            doc["rerank_score"] = score
            scored_docs.append(doc)
        return sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)

