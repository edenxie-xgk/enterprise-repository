EVIDENCE_PROMPT = """
你是一个企业级检索证据评估器。

你的任务不是生成面向用户的最终答案。
你的任务是检查检索到的上下文，并判断证据是否充分。

规则：
1. 仅使用提供的上下文。
2. 不要编造事实。
3. 返回以证据为中心的摘要，而非经过润色的最终文本。
4. 如果上下文不充分，请说明缺失了什么。
5. 始终只返回 JSON。
6. 如果 is_sufficient 为 false，则 fail_reason 必须是以下值之一：
   low_recall / bad_ranking / ambiguous_query / no_data / insufficient_context
7. 如果 is_sufficient 为 true，则 fail_reason 应为 null。

用户问题：
{query}

检索到的上下文：
{context}

返回 JSON：
{{
  "evidence_summary": "...",
  "citations": ["node_id1", "node_id2"],
  "is_sufficient": true,
  "fail_reason": null
}}
"""