from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from utils.logger_handler import logger
from core.settings import settings
from src.congfig.llm_config import LLMService
from src.prompts.rag.qa_generation import QA_GENERATION_PROMPT


class QaData(BaseModel):
    question:str = Field(...,description="问题")
    answer:str = Field(...,description="答案")
    language:str = Field(...,description="语言")
    difficulty:str = Field(...,description="难度")
    intent:str = Field(...,description="意图")

class  QAResult(BaseModel):
    """QA数据构建助手返回的数据"""
    qa_list:List[QaData] = Field(...,description="QA列表")

def generate_qa(llm:BaseChatModel, chunk, node_id,metadata):

    prompt = QA_GENERATION_PROMPT.format(chunk=chunk)

    for i in range(settings.max_retries):
        try:
            llm = llm.with_structured_output(QAResult)
            response = llm.invoke([HumanMessage(content=prompt)])
            qa_list = []
            for item in response.qa_list:
                if not item.question or  not item.answer:
                    continue
                qa_list.append({
                    "question": item.question,
                    "answer": item.answer,
                    "language": item.language,
                    "difficulty": item.difficulty,
                    "intent": item.intent,
                    "node_id": node_id,
                    "metadata": metadata
                })
            return qa_list
        except Exception as e:
            logger.warning(f"[LLM失败] 第{i + 1}次: {e}")
            pass
    logger.error(f"QA构建数据失败：f{chunk}")
    return []
