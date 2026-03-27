from anthropic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from src.congfig.llm_config import LLMService
from src.prompts.rag.generation_prompt import GEN_PROMPT
from src.rag.generation.answer_verify import verify_answer
from src.rag.generation.translate import translate


class GeneratorResult(BaseModel):
    """
        问答系统返回的数据
    """
    answer: str = Field(...,description="回答内容")
    citations: list[str] = Field(...,description="引用的node编号")

def generate_answer (llm:BaseChatModel, query: str, context: str):
        prompt = GEN_PROMPT.format(
            query=query,
            context=context
        )

        response:GeneratorResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=GeneratorResult
        )

        return  response

