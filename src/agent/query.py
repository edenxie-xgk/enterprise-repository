from typing import List, Dict

from pydantic import BaseModel



class QueryAgentOutput(BaseModel):
    rewrite_query: str
    search_queries: List[str]
    intent: str
    filters: Dict[str, str]
    confidence: float
    search_queries:List[str]


class QueryAgent:
    def __init__(self, llm):
        self.llm = llm




query_agent = QueryAgent(llm=None)