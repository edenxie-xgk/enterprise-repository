from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

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


class ChatGPTResult(BaseModel):
    answer: str

if __name__ == "__main__":
    chatgpt_llm = chatgpt_llm.with_structured_output(ChatGPTResult,include_raw=True)
    prompt = """
        请你一句话回答什么是机器学习
        
        【输出（必须是JSON）】
        {
          "answer": "..."
        }
    
    """
    res = chatgpt_llm.invoke([HumanMessage(content=prompt)])
    print(res['parsed'])
