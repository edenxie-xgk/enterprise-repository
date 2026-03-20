from typing import List, Literal, Dict

from pydantic import BaseModel, Field


class QueryResult(BaseModel):
    """重写查询的输出数据"""
    rewrite_query:str = Field(...,description="重写后的查询")
    search_queries:List[str] = Field(...,description="重写后的查询多语言版，每条数据对应一种语言")
    filters:Dict = Field(...,description="过滤条件")
    intent: Literal["factoid", "analysis", "comparison"] = Field(..., description="意图")