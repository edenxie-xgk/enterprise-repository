import json
from typing import Literal

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
    is_sufficient:bool = Field(...,description="是否能够回答问题")
    fail_reason:  Literal[
        "low_recall",      # 没召回
        "bad_ranking",     # 排序差
        "ambiguous_query", # query不清晰
        "no_data",         # 没数据
        "nothing_abnormal", #没异常
    ] = Field(default=None,description="诊断信息")

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

if  __name__ == "__main__":
    a = '{\n  "answer": "根据当前资料无法回答该问题",\n  "citations": [],\n  "is_sufficient": false,\n  "fail_reason": "no_data"\n}'
    print(GeneratorResult(**json.loads(a)))