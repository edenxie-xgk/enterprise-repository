# Scripts 目录说明

`scripts/` 目录存放的是这个项目当前可直接运行的辅助脚本。

建议从项目根目录执行：

```bash
python scripts/<script_name>.py
```

部分脚本依赖 `.env` 中的数据库配置，LoRA 相关脚本还依赖额外训练依赖。

## 脚本总览

| 脚本 | 作用 |
| --- | --- |
| `init_project.py` | 初始化数据库结构并导入启动种子数据 |
| `export_db_exports.py` | 把当前 PostgreSQL / MongoDB 数据导出到 `db/` 目录 |
| `import_db_exports.py` | 用 `db/` 目录里的导出文件覆盖当前 PostgreSQL / MongoDB |
| `generate_qa_dataset.py` | 从已入库文档生成 QA 数据，或批量回滚文档 QA 状态 |
| `export_financial_fact_lora.py` | 从金融事实图库导出 LoRA 训练样本 |
| `prepare_financial_fact_lora_from_hf.py` | 从 Hugging Face 财报数据集构建 LoRA 训练样本 |
| `train_financial_fact_extractor.py` | 训练金融事实抽取 LoRA 适配器 |

## 1. 初始化项目

### `init_project.py`

用于初始化数据库结构，并按需导入启动种子数据。

```bash
python scripts/init_project.py
python scripts/init_project.py --schema-only
python scripts/init_project.py --seed-only
python scripts/init_project.py --seed-file db/seed/bootstrap_seed.example.json
```

主要参数：

- `--mode`：schema 初始化模式，可选 `auto`、`migrate`、`create_all`、`none`
- `--schema-only`：只初始化 schema，不导入种子数据
- `--seed-only`：只导入种子数据，要求 schema 已存在
- `--seed-file`：指定种子文件路径

脚本最终会输出一份 JSON 摘要。

## 2. 数据库导出与导入

### `export_db_exports.py`

把当前数据库内容导出到项目内的 `db/postgre/` 和 `db/mongodb/`。

当前导出的 PostgreSQL 表：

- `data_rag_doc`
- `department`
- `file`
- `role`
- `role_department`
- `users`

当前导出的 MongoDB 集合：

- `rag_doc`
- `rag_qa`

```bash
python scripts/export_db_exports.py
python scripts/export_db_exports.py --postgres-only
python scripts/export_db_exports.py --mongo-only
python scripts/export_db_exports.py --overwrite
```

主要参数：

- `--postgres-only`：只导出 PostgreSQL
- `--mongo-only`：只导出 MongoDB
- `--overwrite`：覆盖已有导出文件；不指定时会写入带时间戳的新文件

### `import_db_exports.py`

把 `db/postgre/` 和 `db/mongodb/` 目录中的导出文件重新导入当前数据库。

```bash
python scripts/import_db_exports.py
python scripts/import_db_exports.py --postgres-only
python scripts/import_db_exports.py --mongo-only
```

主要参数：

- `--postgres-only`：只导入 PostgreSQL
- `--mongo-only`：只导入 MongoDB

说明：

- PostgreSQL 导入前会对目标表执行 `truncate ... restart identity cascade`
- MongoDB 导入前会清空目标集合再写入
- 这两个脚本都依赖 `.env` 中的数据库连接配置

## 3. QA 数据生成

### `generate_qa_dataset.py`

用于从已入库的 RAG 文档生成 QA 数据，也支持把 QA 源文档状态批量回滚。

### 生成 QA

```bash
python scripts/generate_qa_dataset.py
python scripts/generate_qa_dataset.py --limit 20 --dry-run
python scripts/generate_qa_dataset.py --department-id 1 --export-path data/rag_agent.rag_qa.json
```

常用参数：

- `--source-state`：只处理指定状态的源文档，默认 `2`
- `--mark-state`：处理成功后把源文档更新到该状态，默认 `1`
- `--limit`：限制处理文档数量
- `--department-id`：按部门过滤
- `--dense-score-threshold`：相关文档最小稠密检索分数
- `--dense-top-k`：相关文档候选数量
- `--max-related-docs`：每个源文档保留的相关文档数
- `--max-qa-per-doc`：每个源文档最多保留多少条 QA，默认 `2`
- `--dry-run`：只生成和汇总，不写回数据库
- `--export-path`：额外导出本次生成的 QA JSON

默认启用严格校验，相关参数如下：

- `--disable-verification`：关闭生成后校验
- `--verification-retrieval-top-k`：校验阶段的检索候选数量
- `--verification-rerank-top-k`：校验阶段的重排保留数量
- `--verification-answer-score-threshold`：答案回归一致性阈值
- `--allow-missing-retrieval-coverage`：校验时允许检索没覆盖全部声明节点
- `--allow-missing-rerank-coverage`：校验时允许 rerank 没覆盖全部声明节点

### 回滚 QA 源文档状态

```bash
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 1 --rollback-to-state 2 --dry-run
python scripts/generate_qa_dataset.py --rollback-all --rollback-from-state 1 --rollback-to-state 2
```

回滚参数：

- `--rollback-all`：启用批量回滚模式
- `--rollback-from-state`：当前状态筛选，默认 `1`
- `--rollback-to-state`：目标状态，默认 `2`
- `--dry-run`：只预览，不实际更新
- `--export-path`：导出回滚结果摘要

## 4. LoRA 数据准备

### `export_financial_fact_lora.py`

从金融事实图库中读取最近的事实数据，并按 `node_id` 聚合后导出为 LoRA 训练 JSONL。

```bash
python scripts/export_financial_fact_lora.py
python scripts/export_financial_fact_lora.py --output data/financial_fact_lora.jsonl --limit 500
```

主要参数：

- `--output`：输出 JSONL 路径，默认 `data/financial_fact_lora.jsonl`
- `--limit`：最多读取多少条 fact 记录，默认 `500`

说明：

- 依赖 `GRAPH_FACT_COLLECTION_NAME`
- 输出目录会自动创建

### `prepare_financial_fact_lora_from_hf.py`

从 Hugging Face 数据集读取财报文本或 PDF，切块后调用金融事实抽取逻辑，导出 LoRA 训练 JSONL。

```bash
python scripts/prepare_financial_fact_lora_from_hf.py --dataset ranzaka/cse_financial_reports --max-documents 100
```

主要参数：

- `--dataset`：Hugging Face 数据集名称，默认 `ranzaka/cse_financial_reports`
- `--split`：数据集 split，默认 `train`
- `--output`：输出 JSONL 路径，默认 `data/financial_fact_lora_from_hf.jsonl`
- `--cache-dir`：Hugging Face 缓存目录
- `--streaming`：启用流式读取
- `--max-documents`：最多处理多少个源文档
- `--max-pages`：每个 PDF 最多抽取多少页
- `--max-chars`：每个文档最多读取多少字符
- `--chunk-size`：切块大小
- `--chunk-overlap`：切块重叠字符数
- `--min-chars`：最小 chunk 长度
- `--max-facts-per-chunk`：每个 chunk 最多保留多少条事实

说明：

- 依赖 `datasets` 包
- 输出内容是 JSONL，每行一条训练样本

## 5. LoRA 训练

### `train_financial_fact_extractor.py`

使用 JSONL 训练集训练金融事实抽取 LoRA 适配器。

```bash
python scripts/train_financial_fact_extractor.py --model-name Qwen/Qwen2.5-7B-Instruct
python scripts/train_financial_fact_extractor.py --model-name Qwen/Qwen2.5-7B-Instruct --train-file data/financial_fact_lora_from_hf.jsonl --output-dir outputs/financial_fact_extractor_lora
```

主要参数：

- `--train-file`：训练数据文件，默认 `data/financial_fact_lora_from_hf.jsonl`
- `--model-name`：基础模型名称，必填
- `--output-dir`：适配器输出目录，默认 `outputs/financial_fact_extractor_lora`
- `--max-length`：最大序列长度
- `--batch-size`：单步 batch size
- `--gradient-accumulation-steps`：梯度累积步数
- `--num-train-epochs`：训练轮数
- `--learning-rate`：学习率
- `--max-steps`：最大优化步数，`0` 表示不限制
- `--log-every`：每多少个 optimizer step 打印一次 loss
- `--dtype`：权重精度，可选 `fp32`、`fp16`、`bf16`
- `--lora-r`：LoRA rank
- `--lora-alpha`：LoRA alpha
- `--lora-dropout`：LoRA dropout
- `--lora-target-modules`：LoRA 注入模块，逗号分隔
- `--trust-remote-code`：允许加载远程模型代码

训练完成后会输出：

- LoRA adapter 权重
- tokenizer 文件
- `training_summary.json`

LoRA 相关依赖可通过下面命令安装：

```bash
pip install -r requirements-train.txt
```
