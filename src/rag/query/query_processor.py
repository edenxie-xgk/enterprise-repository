from typing import List, Dict, Literal

from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from src.models.llm import deepseek_llm
from src.rag.models import QueryResult
from src.rag.query.normalize import normalize_query
from src.rag.query.rewrite import rewrite





class QueryProcessor:

    def __init__(self,llm:BaseChatOpenAI):
        self.llm = llm


    def run(self, query, chat_history=None, user_profile=None):
        """清洗输入、重写输入"""

        #清洗输入
        query = normalize_query(query)

        #重写输入
        result: QueryResult = rewrite(self.llm, query, chat_history)


        return result
query_processor = QueryProcessor(deepseek_llm)


if  __name__ == "__main__":
    res = query_processor.run("请你帮我分析一下永动机的是否可以创造出来？")
    print(res)
    print(res.search_queries)

