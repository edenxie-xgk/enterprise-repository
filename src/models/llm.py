from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI

from core.settings import settings

deepseek_llm =  ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=3,
    api_key=settings.deepseek_api_key,
)

chatgpt_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)

if __name__ == "__main__":
    res = chatgpt_llm.invoke("请你一句话回答什么是机器学习")
    print(res)