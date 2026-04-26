# 📘 Scripts 目录说明

这个目录用于集中存放当前项目可直接使用的脚本：

- 🧱 初始化数据库与种子数据
- 🧪 生成 QA 数据，便于做检索链路验收或人工抽查
- 🧠 准备 LoRA 训练数据
- 🚀 训练金融事实抽取 LoRA 适配器

## 🗂️ 脚本总览

| 脚本 | 作用 |
| --- | --- |
| `init_project.py` | 初始化数据库结构与基础数据 |
| `generate_qa_dataset.py` | 从已入库文档生成 QA 数据 |
| `export_financial_fact_lora.py` | 从图谱事实导出 LoRA 训练样本 |
| `prepare_financial_fact_lora_from_hf.py` | 从 Hugging Face 财报数据构建 LoRA 训练样本 |
| `train_financial_fact_extractor.py` | 训练金融事实抽取 LoRA 适配器 |

## 🧭 推荐流程

### 1. 初始化项目

用于初始化数据库结构和基础数据。

```bash
python scripts/init_project.py
python scripts/init_project.py --schema-only
python scripts/init_project.py --seed-file db/seed/bootstrap_seed.example.json
```

### 2. 生成 QA 数据

把已入库的 RAG 文档转换成 QA 数据，适合做检索链路验收、抽样检查或后续 benchmark 数据准备。

```bash
python scripts/generate_qa_dataset.py
python scripts/generate_qa_dataset.py --limit 20 --dry-run
python scripts/generate_qa_dataset.py --department-id 1 --export-path data/rag_agent.rag_qa.json
```

常用参数：

- `--source-state`：待处理源文档状态，默认 `2`
- `--mark-state`：处理完成后回写的状态，默认 `1`
- `--limit`：限制本次处理文档数量
- `--department-id`：按部门过滤
- `--dense-score-threshold`：相关文档最小相似度阈值
- `--dense-top-k`：相关文档候选数量
- `--max-related-docs`：每个源文档保留的关联文档数
- `--dry-run`：只生成，不写回数据库
- `--export-path`：把本次生成的 QA 额外导出为 JSON

### 3. 回滚文档状态

`generate_qa_dataset.py` 也支持把全部 QA 源文档状态统一回滚，适合处理误标记、补跑 QA 生成等场景。

先预览全部待回滚文档：

```bash
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 1 --rollback-to-state 2 --dry-run
```

正式执行全部回滚：

```bash
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 1 --rollback-to-state 2
```

如果你的状态定义不同，也可以自定义起始状态和目标状态：

```bash
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 3 --rollback-to-state 2
```

回滚相关参数：

- `--rollback-all`：启用全部回滚模式
- `--rollback-from-state`：当前状态筛选，默认 `1`
- `--rollback-to-state`：目标状态，默认 `2`
- `--dry-run`：只预览命中的文档，不实际更新状态
- `--export-path`：导出回滚摘要

## 🧠 LoRA 数据准备

训练数据可以从两个入口准备：

### 方案 A：从图谱事实导出

适合已经完成图谱抽取，希望直接基于现有事实构造训练样本。

```bash
python scripts/export_financial_fact_lora.py --output data/financial_fact_lora.jsonl --limit 500
```

默认输出：

```text
data/financial_fact_lora.jsonl
```

### 方案 B：从 Hugging Face 财报数据准备

适合想补充公开语料，快速扩充 LoRA 训练样本来源。

```bash
python scripts/prepare_financial_fact_lora_from_hf.py --dataset ranzaka/cse_financial_reports --max-documents 100
```

默认输出：

```text
data/financial_fact_lora_from_hf.jsonl
```

## 🚀 LoRA 训练

`train_financial_fact_extractor.py` 用于训练金融事实抽取 LoRA 适配器。

```bash
python scripts/train_financial_fact_extractor.py --model-name Qwen/Qwen2.5-7B-Instruct
```

指定训练集和输出目录：

```bash
python scripts/train_financial_fact_extractor.py ^
  --model-name Qwen/Qwen2.5-7B-Instruct ^
  --train-file data/financial_fact_lora_from_hf.jsonl ^
  --output-dir outputs/financial_fact_extractor_lora
```

常用参数：

- `--train-file`：LoRA 训练数据文件，默认 `data/financial_fact_lora_from_hf.jsonl`
- `--model-name`：基础模型名，必填
- `--output-dir`：LoRA 适配器输出目录
- `--lora-r`：LoRA rank
- `--lora-alpha`：LoRA alpha
- `--lora-dropout`：LoRA dropout
- `--dtype`：权重精度，可选 `fp32` / `fp16` / `bf16`

训练完成后会输出：

- LoRA 适配器权重
- tokenizer 文件
- `training_summary.json`

## 📦 训练依赖

LoRA 训练相关补充依赖在 `requirements-train.txt`：

```bash
pip install -r requirements-train.txt
```

## ✅ 使用建议

- 第一次跑训练前，建议先用小样本验证整条链路
- 如果只是验证数据准备逻辑，优先加 `--limit`
- 训练产物建议统一放到 `outputs/financial_fact_extractor_lora/`
