from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Callable, Iterator


AnswerTokenHandler = Callable[[str], None]

_answer_token_handler: ContextVar[AnswerTokenHandler | None] = ContextVar(
    "answer_token_handler",
    default=None,
)


def get_answer_token_handler() -> AnswerTokenHandler | None:
    return _answer_token_handler.get()


def has_answer_token_handler() -> bool:
    return get_answer_token_handler() is not None


def emit_answer_token(token: str) -> None:
    handler = get_answer_token_handler()
    if handler and token:
        handler(token)


@contextmanager
def bind_answer_token_handler(handler: AnswerTokenHandler | None) -> Iterator[None]:
    token: Token | None = None
    if handler is not None:
        token = _answer_token_handler.set(handler)
    try:
        yield
    finally:
        if token is not None:
            _answer_token_handler.reset(token)
