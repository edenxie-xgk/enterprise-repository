from core.settings import settings


class ContextBuilder:

    def __init__(self):
        pass

    def _deduplicate(self,docs):
        """去重"""
        seen = set()
        results = []

        for doc in docs:
            text = doc["content"]
            if text not in seen:
                seen.add(text)
                results.append(doc)

        return results

    def _truncate_docs(self,docs, max_length=settings.context_max_len):
        """截断（控制长度）"""
        total = 0
        results = []

        for doc in docs:
            text = doc["content"]

            if total + len(text) > max_length:
                break

            results.append(doc)
            total += len(text)

        return results

    def _build_context(self,docs):
        """拼接文档"""
        context_parts = []

        for i, doc in enumerate(docs):
            context_parts.append(
                f"[文档{i+1}]\n{doc['content']}"
            )

        return "\n\n".join(context_parts)


    def run(self,docs):
        # 1. 去重
        docs = self._deduplicate(docs)

        # 2. 截断
        docs = self._truncate_docs(docs)

        # 3. 拼接
        context = self._build_context(docs)

        return context