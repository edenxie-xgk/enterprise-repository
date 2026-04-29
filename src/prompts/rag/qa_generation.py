QA_GENERATION_PROMPT = """
You are preparing a high-precision QA benchmark for an enterprise RAG system.

Task:
Generate QA items only when the answer is strictly grounded in the provided nodes.

Requirements:
1. Precision is more important than recall. If the evidence is weak, ambiguous, repetitive, or low quality, return an empty `qa_list`.
2. Every question must require at least 2 nodes to answer.
3. Every answer must be fully supported by the provided nodes. Do not add assumptions, background knowledge, hidden causes, or unstated time ranges.
4. Prefer factual or comparison questions with short, concrete answers.
5. Only generate an analysis question when the conclusion is explicitly supported by the nodes.
6. Use the same language as the source documents. Target language: `{source_language}`.
7. Do not translate into another language.
8. `node_ids` must contain only the minimal supporting nodes for the answer.
9. If a question can be answered from a single node, do not generate it.
10. If the answer contains a number, date, ratio, entity, or other factual value, that value must appear explicitly in the nodes.
11. At most generate `{max_qa_per_batch}` QA items.
12. `difficulty` must be one of: `easy`, `medium`, `hard`.
13. `intent` must be one of: `factoid`, `comparison`, `analysis`.
14. Output JSON only.

Nodes:
{nodes}

Output JSON:
{{
  "qa_list": [
    {{
      "question": "...",
      "answer": "...",
      "language": "{source_language}",
      "difficulty": "easy",
      "intent": "factoid",
      "node_ids": ["node1", "node2"]
    }}
  ]
}}
"""
