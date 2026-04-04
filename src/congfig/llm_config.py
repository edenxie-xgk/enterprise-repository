import time
from typing import Any, Optional, Type

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from core.settings import settings
from src.models.llm import chatgpt_llm
from utils.logger_handler import logger


class LLMService:
    @staticmethod
    def _extract_usage(response: Any) -> dict:
        if isinstance(response, dict):
            raw = response.get("raw")
            metadata = getattr(raw, "response_metadata", {}) or {}
            return metadata.get("token_usage", {}) or {}

        metadata = getattr(response, "response_metadata", {}) or {}
        return metadata.get("token_usage", {}) or {}

    @staticmethod
    def _extract_payload(response: Any, schema: Optional[Type[BaseModel]] = None):
        if schema and isinstance(response, dict):
            return response.get("parsed")
        if not schema and isinstance(response, dict):
            return response.get("raw", response)
        return response

    @staticmethod
    def _invoke_once(
        llm: BaseChatModel,
        messages,
        schema: Optional[Type[BaseModel]] = None,
    ):
        start = time.time()
        response = llm.invoke(messages)
        duration = time.time() - start
        if duration > settings.max_timeout:
            logger.warning(f"[LLM超时] 耗时: {duration:.2f}s")
        else:
            logger.info(f"[LLM调用成功] 耗时: {duration:.2f}s")

        usage = LLMService._extract_usage(response)
        logger.info(f"[tokens]: {usage}")
        return LLMService._extract_payload(response, schema=schema)

    @staticmethod
    def invoke(
        llm: BaseChatModel,
        messages,
        schema: Optional[Type[BaseModel]] = None,
        fallback_llm: BaseChatModel = None,
    ):
        last_exception = None
        if schema:
            llm = llm.with_structured_output(schema, include_raw=True)

        for i in range(settings.max_retries):
            try:
                return LLMService._invoke_once(llm, messages, schema=schema)
            except Exception as e:
                last_exception = e
                logger.warning(f"[LLM失败] 第{i + 1}次: {e}")
                time.sleep(2 ** i)

        if not fallback_llm:
            fallback_llm = chatgpt_llm

        if schema:
            fallback_llm = fallback_llm.with_structured_output(schema, include_raw=True)

        try:
            return LLMService._invoke_once(fallback_llm, messages, schema=schema)
        except Exception as e:
            last_exception = e
            raise last_exception


