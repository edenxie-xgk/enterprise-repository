import time
from contextvars import ContextVar, Token
from typing import Any, Callable, Optional, Type

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from core.settings import settings
from utils.logger_handler import logger


class LLMService:
    _usage_records: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_usage_records", default=None)

    @staticmethod
    def _load_fallback_llm() -> BaseChatModel:
        from src.models.llm import chatgpt_llm

        return chatgpt_llm

    @staticmethod
    def _resolve_model_name(llm: BaseChatModel) -> str:
        return (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or llm.__class__.__name__
        )

    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
        prompt_tokens = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("input_token_count")
            or 0
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("output_token_count")
            or 0
        )
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _estimate_cost(model_name: str, usage: dict[str, int]) -> float:
        name = (model_name or "").lower()
        input_cost = settings.monitor_default_input_cost_per_1m
        output_cost = settings.monitor_default_output_cost_per_1m

        if "deepseek" in name:
            input_cost = settings.monitor_deepseek_input_cost_per_1m
            output_cost = settings.monitor_deepseek_output_cost_per_1m
        elif "gpt" in name or "openai" in name or name.startswith("o"):
            input_cost = settings.monitor_openai_input_cost_per_1m
            output_cost = settings.monitor_openai_output_cost_per_1m

        return (
            usage["prompt_tokens"] * input_cost
            + usage["completion_tokens"] * output_cost
        ) / 1_000_000

    @classmethod
    def start_usage_collection(cls) -> Token:
        return cls._usage_records.set([])

    @classmethod
    def stop_usage_collection(cls, token: Token) -> list[dict[str, Any]]:
        records = list(cls._usage_records.get() or [])
        cls._usage_records.reset(token)
        return records

    @classmethod
    def summarize_usage(cls, records: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_tokens = sum(int(item.get("prompt_tokens", 0)) for item in records)
        completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in records)
        total_tokens = sum(int(item.get("total_tokens", 0)) for item in records)
        duration_ms = sum(int(item.get("duration_ms", 0)) for item in records)
        estimated_cost = round(sum(float(item.get("estimated_cost_usd", 0)) for item in records), 6)
        models = sorted({str(item.get("model", "")) for item in records if item.get("model")})
        return {
            "call_count": len(records),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "estimated_cost_usd": estimated_cost,
            "models": models,
            "calls": records,
        }

    @classmethod
    def _record_usage(cls, *, model_name: str, usage: dict[str, Any], duration: float) -> None:
        bucket = cls._usage_records.get()
        if bucket is None:
            return

        normalized_usage = cls._normalize_usage(usage)
        bucket.append(
            {
                "model": model_name,
                "prompt_tokens": normalized_usage["prompt_tokens"],
                "completion_tokens": normalized_usage["completion_tokens"],
                "total_tokens": normalized_usage["total_tokens"],
                "duration_ms": int(duration * 1000),
                "estimated_cost_usd": round(cls._estimate_cost(model_name, normalized_usage), 6),
            }
        )

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
    def _extract_stream_text(chunk: Any) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif "text" in item:
                        parts.append(str(item.get("text") or ""))
                    continue
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _extract_stream_usage(chunk: Any) -> dict:
        usage = getattr(chunk, "usage_metadata", None)
        if isinstance(usage, dict):
            return usage

        metadata = getattr(chunk, "response_metadata", {}) or {}
        if isinstance(metadata, dict):
            return metadata.get("token_usage", {}) or {}
        return {}

    @staticmethod
    def _invoke_once(
        llm: BaseChatModel,
        messages,
        schema: Optional[Type[BaseModel]] = None,
        model_name: str = "",
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
        LLMService._record_usage(model_name=model_name, usage=usage, duration=duration)
        return LLMService._extract_payload(response, schema=schema)

    @staticmethod
    def _stream_once(
        llm: BaseChatModel,
        messages,
        *,
        on_token: Callable[[str], None] | None = None,
        model_name: str = "",
    ) -> str:
        start = time.time()
        text_parts: list[str] = []
        usage: dict[str, Any] = {}

        for chunk in llm.stream(messages):
            token_text = LLMService._extract_stream_text(chunk)
            if token_text:
                text_parts.append(token_text)
                if on_token:
                    on_token(token_text)

            chunk_usage = LLMService._extract_stream_usage(chunk)
            if chunk_usage:
                usage = chunk_usage

        duration = time.time() - start
        if duration > settings.max_timeout:
            logger.warning(f"[LLM streaming timeout] duration: {duration:.2f}s")
        else:
            logger.info(f"[LLM streaming success] duration: {duration:.2f}s")

        logger.info(f"[stream tokens]: {usage}")
        LLMService._record_usage(model_name=model_name, usage=usage, duration=duration)
        return "".join(text_parts)

    @staticmethod
    def invoke(
        llm: BaseChatModel,
        messages,
        schema: Optional[Type[BaseModel]] = None,
        fallback_llm: BaseChatModel = None,
    ):
        last_exception = None
        model_name = LLMService._resolve_model_name(llm)
        if schema:
            llm = llm.with_structured_output(schema, include_raw=True)

        for i in range(settings.max_retries):
            try:
                return LLMService._invoke_once(llm, messages, schema=schema, model_name=model_name)
            except Exception as e:
                last_exception = e
                logger.warning(f"[LLM失败] 第{i + 1}次: {e}")
                time.sleep(2 ** i)

        if not fallback_llm:
            fallback_llm = LLMService._load_fallback_llm()

        fallback_model_name = LLMService._resolve_model_name(fallback_llm)
        if schema:
            fallback_llm = fallback_llm.with_structured_output(schema, include_raw=True)

        try:
            return LLMService._invoke_once(
                fallback_llm,
                messages,
                schema=schema,
                model_name=fallback_model_name,
            )
        except Exception as e:
            last_exception = e
            raise last_exception

    @staticmethod
    def stream_text(
        llm: BaseChatModel,
        messages,
        *,
        on_token: Callable[[str], None] | None = None,
        fallback_llm: BaseChatModel = None,
    ) -> str:
        last_exception = None
        model_name = LLMService._resolve_model_name(llm)

        for i in range(settings.max_retries):
            emitted = False

            def emit(token: str) -> None:
                nonlocal emitted
                emitted = True
                if on_token:
                    on_token(token)

            try:
                return LLMService._stream_once(
                    llm,
                    messages,
                    on_token=emit,
                    model_name=model_name,
                )
            except Exception as e:
                last_exception = e
                if emitted:
                    logger.warning(f"[LLM streaming failed after emission] {e}")
                    raise
                logger.warning(f"[LLM streaming failed] retry={i + 1}: {e}")
                time.sleep(2 ** i)

        if not fallback_llm:
            fallback_llm = LLMService._load_fallback_llm()

        fallback_model_name = LLMService._resolve_model_name(fallback_llm)
        emitted = False

        def emit(token: str) -> None:
            nonlocal emitted
            emitted = True
            if on_token:
                on_token(token)

        try:
            return LLMService._stream_once(
                fallback_llm,
                messages,
                on_token=emit,
                model_name=fallback_model_name,
            )
        except Exception as e:
            last_exception = e
            if emitted:
                logger.warning(f"[LLM fallback streaming failed after emission] {e}")
            raise last_exception


