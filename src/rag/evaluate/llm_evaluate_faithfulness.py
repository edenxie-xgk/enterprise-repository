from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.config.llm_config import LLMService
from src.prompts.rag.evaluate import FAITHFULNESS_PROMPT
from utils.logger_handler import logger


class EvaluateFaithfulnessResult(BaseModel):
    """模型生成答案的忠实度评分结果"""

    score: float = Field(..., description="忠实度评分")


def evaluate_faithfulness(llm_service: BaseChatModel, query, context, answer):
    prompt = FAITHFULNESS_PROMPT.format(
        query=query,
        context=context,
        answer=answer,
    )

    result: EvaluateFaithfulnessResult = LLMService.invoke(
        llm=llm_service,
        messages=[HumanMessage(content=prompt)],
        schema=EvaluateFaithfulnessResult,
    )

    if result is None:
        logger.warning("[benchmark] faithfulness judge returned empty structured payload")
        return 0.0

    score = getattr(result, "score", None)
    if score is None:
        logger.warning("[benchmark] faithfulness judge payload missing score")
        return 0.0

    return float(score)
