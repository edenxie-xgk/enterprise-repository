from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.congfig.llm_config import LLMService
from src.prompts.agent.translate import TRANSLATE_PROMPT


def translate(llm:BaseChatModel, query: str):
    prompt = TRANSLATE_PROMPT.format(
        query=query
    )
    return LLMService.invoke(
        llm=llm,
        messages=[HumanMessage(content=prompt)],
    ).content