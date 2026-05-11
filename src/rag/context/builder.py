import re
from typing import Any

from core.settings import settings


_WHITESPACE_PATTERN = re.compile(r"\s+")


class ContextBuilder:
    """Build the traditional RAG context consumed by generation."""

    def __init__(self, max_length: int | None = None):
        self.max_length = int(max_length or settings.context_max_len)

    def _get_value(self, item: Any, key: str, default: Any = "") -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _normalize_content_key(self, content: str) -> str:
        return _WHITESPACE_PATTERN.sub(" ", content or "").strip()

    def _deduplicate(self, docs):
        """Deduplicate by node_id first, then normalized content."""
        seen_node_ids = set()
        seen_content = set()
        results = []

        for doc in docs or []:
            node_id = str(self._get_value(doc, "node_id", "") or "").strip()
            content = str(self._get_value(doc, "content", "") or "").strip()
            content_key = self._normalize_content_key(content)

            if node_id:
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)

            if content_key:
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)

            results.append(doc)

        return results

    def _format_doc(self, doc: Any, content_override: str | None = None) -> str:
        node_id = str(self._get_value(doc, "node_id", "") or "").strip()
        content = str(
            self._get_value(doc, "content", "") if content_override is None else content_override
        ).strip()
        return f"[node_id:{node_id}]\n{content}"

    def _truncate_docs(self, docs, max_length: int | None = None):
        """Keep blocks in rank order while accounting for node_id headers."""
        limit = int(max_length or self.max_length)
        total = 0
        results = []

        for doc in docs or []:
            block = self._format_doc(doc)
            separator_length = 2 if results else 0
            block_length = len(block) + separator_length

            if total + block_length <= limit:
                results.append(doc)
                total += block_length
                continue

            if results:
                break

            content = str(self._get_value(doc, "content", "") or "").strip()
            available = limit - len(self._format_doc(doc, content_override="")) - separator_length
            if available <= 0:
                break

            trimmed_content = content[:available].rstrip()
            if trimmed_content:
                results.append(self._copy_with_content(doc, trimmed_content))
            break

        return results

    def _copy_with_content(self, doc: Any, content: str):
        if isinstance(doc, dict):
            copied = dict(doc)
            copied["content"] = content
            return copied
        if hasattr(doc, "model_copy"):
            return doc.model_copy(update={"content": content})
        if hasattr(doc, "copy"):
            return doc.copy(update={"content": content})
        return {"node_id": self._get_value(doc, "node_id", ""), "content": content}

    def _expand_context(self, docs, all_chunks, window=1):
        return docs

    def _build_context(self, docs):
        context_parts = []

        for doc in docs or []:
            context_parts.append(self._format_doc(doc))

        return "\n\n".join(context_parts)

    def run(self, docs):
        docs = self._deduplicate(docs)
        docs = self._truncate_docs(docs)
        return self._build_context(docs)
