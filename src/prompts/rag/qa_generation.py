QA_GENERATION_PROMPT = """
你是企业级RAG系统的数据构建助手。

任务：
基于多个文本片段，生成需要“综合多个片段”才能回答的问题。

---

【要求】

1.  问题必须依赖多个node才能回答
2.  答案必须来自这些node
3.  标注涉及的node_id
4.  不要生成单node即可回答的问题
5.  每个问题独立成立
6.  识别文本的语言（简体中文->language: zh-cn, 英文->language: en）
7.  同种语言最多生成3个QA
8.  至少必须同时生成英文版本和中文版本
9.  对问题进行分级（easy/medium/hard）
10. 对问题进行意图分类（factoid/analysis/comparison）
11. 参考文档质量差可以不生成数据，直接返回JSON空数组
12. 输出格式必须是JSON格式

---

【字段说明】

- question：问题
- answer：答案
- language：语言
- difficulty：分级（easy/medium/hard）
- intent：意图分类（factoid/analysis/comparison）

---

【intent（意图分类）】

必须选择一个：
- factoid（事实查询）
- analysis（分析）
- comparison（对比）

---

【difficulty（分级）】

必须选择一个：
- easy
- medium 
- hard

---


【多个片段内容】
{nodes}

---

【输出格式（JSON）】
[
  {{
    "question": "...",
    "answer": "...",
    "language": "...",
    "difficulty": "...",
    "intent": "...",
    "node_ids": ["node1", "node2"]
  }}
]

只输出JSON，不要解释。
"""