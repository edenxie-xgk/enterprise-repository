from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.config.llm_config import LLMService
from src.prompts.rag.evaluate import EVAL_PROMPT


class EvaluateAnswerResult(BaseModel):
    """模型生成的数据进行评分返回的数据"""
    score: float = Field(...,description="评分")

def evaluate_answer(llm_service:BaseChatModel, query, gt, answer):

    prompt = EVAL_PROMPT.format(
        query=query,
        ground_truth=gt,
        answer=answer
    )

    result:EvaluateAnswerResult = LLMService.invoke(
        llm=llm_service,
        messages=[HumanMessage(content=prompt)],
        schema=EvaluateAnswerResult
    )

    return  result.score
