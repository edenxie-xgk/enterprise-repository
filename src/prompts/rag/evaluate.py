EVAL_PROMPT = """
请判断以下回答进行评分。

【问题】
{query}

【标准答案】
{ground_truth}

【模型回答】
{answer}

---

评估标准：
1. 是否语义正确
2. 是否覆盖关键信息
3. 允许表达不同
4. 输出格式必须是JSON格式

---

输出JSON：
{
  "score": 0-1
}
"""