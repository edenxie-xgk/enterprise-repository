from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.prompts.rag.generation_prompt import GEN_PROMPT


class Generator:

    def __init__(self, llm:BaseChatModel):
        self.llm = llm

    def run(self, query: str, context: str):

        prompt = GEN_PROMPT.format(
            query=query,
            context=context
        )

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content