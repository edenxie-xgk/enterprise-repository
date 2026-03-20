import json

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI

from src.models.llm import deepseek_llm
from src.prompts.rag.rewrite_prompt import REWRITE_PROMPT
from src.rag.models import QueryResult


def rewrite(llm:BaseChatOpenAI, query: str, chat_history=None):

    prompt = REWRITE_PROMPT.format(
        query=query,
        chat_history=chat_history or []
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content
        data = json.loads(content)

        rewrite_query = data.get("rewrite_query", "")
        english_query = data.get("english_query", "")
        intent = data.get("intent", "factoid")

        search_queries = [rewrite_query]

        if english_query:
            search_queries.append(english_query)

        return QueryResult(
            rewrite_query=rewrite_query,
            search_queries=list(set(search_queries)),
            filters={},
            intent=intent
        )

    except Exception:
        # fallback（非常关键）
        return QueryResult(
            rewrite_query=query,
            search_queries=[query],
            filters={},
            intent="factoid"
        )



if __name__ == "__main__":
    print(rewrite(deepseek_llm, "请问你是谁"))