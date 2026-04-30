# 🧰 Scripts 目录说明

`scripts/` 目录存放项目当前可直接运行的辅助脚本。建议从项目根目录执行：

```bash
python scripts/<script_name>.py
```

部分脚本依赖 `.env` 中的数据库和模型配置；LoRA 训练脚本还依赖 `requirements-train.txt` 中的训练扩展依赖。

## 📋 脚本总览

| 图标 | 脚本 | 用途 |
| --- | --- | --- |
| 🧱 | `init_project.py` | 初始化数据库结构并导入启动种子数据 |
| 📤 | `export_db_exports.py` | 将当前 PostgreSQL / MongoDB 数据导出到 `db/` 目录 |
| 📥 | `import_db_exports.py` | 从 `db/` 目录导入 PostgreSQL / MongoDB 数据 |
| ❓ | `generate_qa_dataset.py` | 从已入库文档生成 QA 数据，或批量回滚文档 QA 状态 |
| 🧾 | `export_financial_fact_lora.py` | 从金融事实图谱导出 LoRA 训练样本 |
| 📁 | `prepare_financial_fact_lora_from_data.py` | 从本地 `data/` 目录构建金融事实 LoRA 训练样本 |
| 🧠 | `train_financial_fact_extractor.py` | 训练金融事实抽取 LoRA 适配器 |

## 🧱 1. 初始化项目

### `init_project.py`

用于初始化数据库结构，并按需导入启动种子数据。

```bash
python scripts/init_project.py
python scripts/init_project.py --schema-only
python scripts/init_project.py --seed-only
python scripts/init_project.py --seed-file db/seed/bootstrap_seed.example.json
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `--mode` | Schema 初始化模式，可选 `auto`、`migrate`、`create_all`、`none` |
| `--schema-only` | 只初始化 schema，不导入种子数据 |
| `--seed-only` | 只导入种子数据，要求 schema 已存在 |
| `--seed-file` | 指定种子数据 JSON 文件路径 |

脚本最终会输出一份 JSON 摘要。

## 🗄️ 2. 数据库导出与导入

### `export_db_exports.py`

将当前数据库内容导出到项目内的 `db/postgre/` 和 `db/mongodb/`。

当前 PostgreSQL 导出表：

- `data_rag_doc`
- `department`
- `file`
- `role`
- `role_department`
- `users`

当前 MongoDB 导出集合：

- `rag_doc`
- `rag_qa`

```bash
python scripts/export_db_exports.py
python scripts/export_db_exports.py --postgres-only
python scripts/export_db_exports.py --mongo-only
python scripts/export_db_exports.py --overwrite
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `--postgres-only` | 只导出 PostgreSQL |
| `--mongo-only` | 只导出 MongoDB |
| `--overwrite` | 覆盖已有导出文件；未指定时会写入带时间戳的新文件 |

### `import_db_exports.py`

从 `db/postgre/` 和 `db/mongodb/` 目录重新导入数据。

```bash
python scripts/import_db_exports.py
python scripts/import_db_exports.py --postgres-only
python scripts/import_db_exports.py --mongo-only
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `--postgres-only` | 只导入 PostgreSQL |
| `--mongo-only` | 只导入 MongoDB |

说明：

- PostgreSQL 导入前会对目标表执行 `truncate ... restart identity cascade`。
- MongoDB 导入前会清空目标集合再写入。
- 两个脚本都依赖 `.env` 中的数据库连接配置。

## ❓ 3. QA 数据生成

### `generate_qa_dataset.py`

用于从已入库的 RAG 文档生成 QA 数据，也支持批量回滚 QA 源文档状态。

生成 QA：

```bash
python scripts/generate_qa_dataset.py
python scripts/generate_qa_dataset.py --limit 20 --dry-run
python scripts/generate_qa_dataset.py --department-id 1 --export-path data/rag_agent.rag_qa.json
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--source-state` | 只处理指定状态的源文档，默认 `2` |
| `--mark-state` | 处理成功后将源文档更新到该状态，默认 `1` |
| `--limit` | 限制处理文档数量 |
| `--department-id` | 按部门过滤 |
| `--dense-score-threshold` | 相关文档最小稠密检索分数 |
| `--dense-top-k` | 相关文档候选数量 |
| `--max-related-docs` | 每个源文档保留的相关文档数 |
| `--max-qa-per-doc` | 每个源文档最多保留多少条 QA，默认 `2` |
| `--dry-run` | 只生成和汇总，不写回数据库 |
| `--export-path` | 额外导出本次生成的 QA JSON |

默认启用生成后校验：

| 参数 | 作用 |
| --- | --- |
| `--disable-verification` | 关闭生成后校验 |
| `--verification-retrieval-top-k` | 校验阶段的检索候选数量 |
| `--verification-rerank-top-k` | 校验阶段的重排保留数量 |
| `--verification-answer-score-threshold` | 答案回归一致性阈值 |
| `--allow-missing-retrieval-coverage` | 校验时允许检索未覆盖全部声明节点 |
| `--allow-missing-rerank-coverage` | 校验时允许 rerank 未覆盖全部声明节点 |

回滚 QA 源文档状态：

```bash
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 1 --rollback-to-state 2 --dry-run
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 1 --rollback-to-state 2
```

## 🧾 4. LoRA 数据准备

### `export_financial_fact_lora.py`

从当前金融事实图谱集合中读取事实数据，并按 `node_id` 聚合后导出为 LoRA 训练 JSONL。

```bash
python scripts/export_financial_fact_lora.py
python scripts/export_financial_fact_lora.py --output data/financial_fact_lora.jsonl --limit 500
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `--output` | 输出 JSONL 路径，默认 `data/financial_fact_lora.jsonl` |
| `--limit` | 最多读取多少条 fact 记录，默认 `500` |

说明：

- 依赖 `GRAPH_FACT_COLLECTION_NAME`。
- 输出目录会自动创建。

### `prepare_financial_fact_lora_from_data.py`

从本地 `data/` 目录读取财报文件，抽取文本、切块、执行金融事实抽取，并导出 LoRA 训练 JSONL。

默认读取：

```text
data/chinese_documents_seed/
```

默认输出：

```text
data/financial_fact_lora_from_data.jsonl
```

运行示例：

```bash
python scripts/prepare_financial_fact_lora_from_data.py
python scripts/prepare_financial_fact_lora_from_data.py --input-dir data/chinese_documents_seed --max-documents 20
python scripts/prepare_financial_fact_lora_from_data.py --patterns "*.pdf,*.txt,*.jsonl" --recursive
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `--input-dir` | 本地输入目录，默认 `data/chinese_documents_seed` |
| `--manifest` | 可选 CSV 清单路径；未指定时读取输入目录下的 `manifest.csv` |
| `--patterns` | 文件匹配模式，逗号分隔，默认 `*.pdf` |
| `--recursive` | 递归读取匹配文件 |
| `--output` | 输出 JSONL 路径，默认 `data/financial_fact_lora_from_data.jsonl` |
| `--max-documents` | 最多读取多少个源文件；小于等于 `0` 表示全部 |
| `--max-pages` | 每个 PDF 最多抽取多少页 |
| `--max-chars` | 每个源文件最多读取多少字符 |
| `--chunk-size` | 每个文本块的字符数 |
| `--chunk-overlap` | 文本块重叠字符数 |
| `--min-chars` | 最小文本块长度 |
| `--max-facts-per-chunk` | 每个文本块最多保留多少条事实 |
| `--department-id` | 写入训练样本 metadata 的部门 ID |
| `--department-name` | 写入训练样本 metadata 的部门名称 |

当前支持的本地输入类型：

- `pdf`
- `txt`
- `md`
- `markdown`
- `csv`
- `json`
- `jsonl`

## 🧠 5. LoRA 训练

### `train_financial_fact_extractor.py`

使用 JSONL 训练集训练金融事实抽取 LoRA 适配器。

```bash
python scripts/train_financial_fact_extractor.py --model-name Qwen/Qwen2.5-7B-Instruct
python scripts/train_financial_fact_extractor.py --model-name Qwen/Qwen2.5-7B-Instruct --train-file data/financial_fact_lora_from_data.jsonl --output-dir outputs/financial_fact_extractor_lora
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `--train-file` | 训练数据文件，默认 `data/financial_fact_lora_from_data.jsonl` |
| `--model-name` | 基座因果语言模型名称或本地路径，必填 |
| `--output-dir` | 适配器输出目录，默认 `outputs/financial_fact_extractor_lora` |
| `--max-length` | 最大序列长度 |
| `--batch-size` | 单步 batch size |
| `--gradient-accumulation-steps` | 梯度累积步数 |
| `--num-train-epochs` | 训练轮数 |
| `--learning-rate` | 学习率 |
| `--max-steps` | 最大优化步数，`0` 表示不限制 |
| `--log-every` | 每多少个 optimizer step 打印一次 loss |
| `--dtype` | 权重精度，可选 `fp32`、`fp16`、`bf16` |
| `--lora-r` | LoRA rank |
| `--lora-alpha` | LoRA alpha |
| `--lora-dropout` | LoRA dropout |
| `--lora-target-modules` | LoRA 注入模块，逗号分隔 |
| `--trust-remote-code` | 允许加载模型自定义代码 |

训练完成后会输出：

- LoRA adapter 权重
- tokenizer 文件
- `training_summary.json`

训练扩展依赖：

```bash
pip install -r requirements-train.txt
```
