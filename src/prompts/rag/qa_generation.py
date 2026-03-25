QA_GENERATION_PROMPT = """
你是企业级RAG系统的数据构建助手。

任务：
基于给定文本，生成高质量问答对（QA）。

---

【要求】

1.  问题必须可以通过该文本检索得到
2.  答案必须完全来自文本（不能编造）
3.  问题要具体，不能模糊
4.  避免“这个、那个”之类指代
5.  每个问题独立成立
6.  识别文本的语言（简体中文->language: zh-cn, 英文->language: en）
7.  同种语言最多生成3个QA
8.  至少必须同时生成英文版本和中文版本
9.  对问题进行分级（easy/medium/hard）
10. 对问题进行意图分类（factoid/analysis/comparison）
11. 输出格式必须是JSON格式

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

【文本】
{chunk}

---

【输出格式（JSON）】
[
  {{
    "question": "...",
    "answer": "...",
    "language": "...",
    "difficulty": "...",
    "intent": "..."
  }}
]

只输出JSON，不要解释。
"""