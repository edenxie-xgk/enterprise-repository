# rag-agent

一个面向企业内部知识问答场景的 Agentic RAG 项目。系统围绕“文档检索 + 智能路由 + 权限控制 + 会话追踪”展开，既能回答知识库问题，也能处理部分结构化业务查询，并在允许时补充 Web 搜索结果。

当前仓库的实际形态是：

- Python 后端：`FastAPI + LangGraph + LangChain + LlamaIndex`
- RAG 检索：`PostgreSQL/PGVector + MongoDB/Elasticsearch + Reranker`
- 前端界面：`Vue 3 + Vite + Element Plus + ECharts`
- 运行模式：支持普通查询、流式聊天、文件上传入库、管理端监控

## 1. 项目定位

这个项目不是单纯的“聊天机器人”，而是一个带有明确业务边界的企业知识问答系统，主要面向以下场景：

- 对企业内部上传文档做检索问答
- 根据用户角色和部门限制知识访问范围
- 对简单问题直接回答，减少不必要的检索成本
- 对复杂问题自动改写、扩展或拆解后再检索
- 对结构化问题直接查业务库，例如“我能访问哪些部门/文件”
- 在用户画像允许的前提下，补充外部 Web 搜索结果
- 记录每次 Agent 运行的 trace、tokens、成本、失败原因和会话信息

## 2. 核心能力

- 多路由回答：支持 `direct_answer`、`rag`、`db_search`、`web_search`
- 多格式文档入库：支持 `txt/docx/md/pdf/xlsx/csv/pptx/json/图片`
- 混合检索：向量检索 + BM25 检索 + RRF 融合
- 重排序：支持 LLM rerank 或 CrossEncoder rerank
- 证据护栏：对检索结果进行引用、来源数、摘要充分性校验
- 权限过滤：可按 `department_id`、`user_id` 等元数据做检索过滤
- 用户画像：支持回答风格、语言、偏好主题、是否显示引用、是否允许联网
- 流式输出：聊天接口通过 SSE 逐段回传答案
- 会话持久化：聊天消息、运行报告、监控数据写入 MongoDB
- 管理端监控：查看请求量、活跃用户、模型分布、失败率、运行详情

## 3. 系统架构

```mermaid
flowchart TD
    A["Vue 前端<br/>聊天页 / 管理页"] --> B["FastAPI 服务层"]
    B --> C["鉴权与用户画像<br/>JWT / User Profile"]
    B --> D["LangGraph Agent"]
    D --> E["Direct Answer"]
    D --> F["RAG"]
    D --> G["DB Search"]
    D --> H["Web Search"]
    F --> I["Dense Retrieval<br/>PGVector"]
    F --> J["BM25 Retrieval<br/>MongoDB 或 Elasticsearch"]
    F --> K["Reranker"]
    B --> L["PostgreSQL<br/>业务表 + 向量表"]
    B --> M["MongoDB<br/>聊天记录 / 文档块 / 评估数据"]
    J --> N["Elasticsearch<br/>可选 BM25 引擎"]
```

### 查询执行主链路

1. 用户在前端发起聊天请求
2. FastAPI 根据 JWT 识别当前用户
3. 系统装配用户画像和可访问部门列表
4. LangGraph 先做 `resolved_query`
5. `agent_node` 根据策略决定下一步动作
6. 动作可能是：
   - `direct_answer`：简单问题直接回答
   - `rag`：企业知识库检索增强回答
   - `db_search`：结构化业务查询
   - `web_search`：外部搜索增强
   - `rewrite_query / expand_query / decompose_query`：中间推理步骤
7. 检索结果进入 `finalize_node` 统一整理为最终回答
8. 运行报告、trace、action_history、消息内容写入 MongoDB
9. 前端通过 SSE 接收流式输出与运行摘要

## 4. 目录结构

```text
rag-agent/
├─ app.py                     # 服务启动入口
├─ core/                      # 全局配置、基础类型
├─ service/                   # FastAPI 路由、SQLModel 模型、鉴权与业务接口
├─ src/                       # Agent、RAG、工具、提示词、状态模型
├─ web_service/               # Vue 3 前端
├─ tests/                     # 单元测试
├─ scripts/                   # 辅助脚本与实验性工具
├─ data/                      # 样例数据、评估数据
├─ db/                        # 本地数据库目录或数据挂载目录
├─ logs/                      # 运行日志
├─ requirements.txt          # Python 依赖
└─ langgraph.json            # LangGraph 图配置
```

### 重点目录说明

#### `core/`

- `settings.py`：项目总配置入口，几乎所有环境变量都从这里读取
- `custom_types.py`：文档元数据模型 `DocumentMetadata`

#### `service/`

- `server.py`：创建 FastAPI 应用、挂载路由、初始化表结构
- `database/`：异步 PostgreSQL 连接、Session 管理、并发控制信号量
- `dependencies/`：鉴权依赖，例如当前登录用户解析
- `models/`：SQLModel 表模型，如用户、文件、角色、部门、用户画像
- `router/`：API 路由
- `utils/chat_store.py`：会话、消息、运行记录持久化
- `utils/user_profile.py`：用户画像读写与序列化

#### `src/`

- `agent/`：LangGraph 图、策略、路由、运行器
- `nodes/`：图节点实现
- `tools/`：对节点调用逻辑的进一步封装
- `rag/`：入库、检索、重排、上下文构造、评估
- `models/`：LLM、Embedding、Reranker 实例化
- `prompts/`：Agent/RAG 提示词模板
- `types/`：状态、事件、结果对象定义
- `database/`：MongoDB、PostgreSQL、Elasticsearch 访问适配

#### `web_service/`

- `src/view/chat/`：聊天主页面
- `src/view/admin/`：管理端监控页面
- `src/components/`：聊天窗、侧栏、用户画像面板等
- `src/api/`：前端请求封装和 SSE 聊天客户端

## 5. 技术栈

| 层次 | 主要技术 |
| --- | --- |
| Web 后端 | FastAPI, Uvicorn |
| Agent 编排 | LangGraph, LangChain |
| RAG 框架 | LlamaIndex |
| LLM 接入 | OpenAI, DeepSeek |
| Web 搜索 | 智谱 `zhipuai` |
| 关系型数据库 | PostgreSQL, SQLModel, SQLAlchemy |
| 向量存储 | PGVector |
| 文档与会话存储 | MongoDB |
| 稀疏检索 | MongoDB BM25 Lite 或 Elasticsearch |
| 重排序 | LLM Reranker / SentenceTransformer CrossEncoder |
| 文档解析 | PyMuPDF, PaddleOCR, python-docx, pandas, python-pptx |
| 前端 | Vue 3, Vite, Element Plus, Tailwind CSS, ECharts |

## 6. 核心模块说明

### 6.1 Agent 编排层

- `src/agent/graph.py`
  - 定义整个 LangGraph 状态图
  - 入口节点是 `resolved_query`
  - 最终通过 `finish` 或 `abort` 结束

- `src/agent/policy.py`
  - 决定输入是否合法
  - 决定初始动作和后续重试动作
  - 根据上一步结果动态调大或调小检索参数
  - 判断何时提前结束或进入 `finalize`

- `src/agent/runner.py`
  - `run_agent()` 是统一调用入口
  - 负责构建初始状态、收集 LLM 使用情况、生成运行报告

### 6.2 节点层

- `resolved_query_node`
  - 结合聊天历史把当前问题归一化成可执行查询

- `agent_node`
  - 图里的决策中枢
  - 根据状态决定继续改写、检索、直答还是结束

- `direct_answer_node`
  - 对简单问题直接回答

- `rewrite_query_node`
  - 对表达含糊、依赖上下文的问题做改写

- `expand_query_node`
  - 生成更多检索变体，提升召回率

- `decompose_query_node`
  - 把复杂问题拆成多个子问题

- `rag_node`
  - 执行企业知识库检索
  - 附带权限过滤、用户偏好主题引导、检索策略调整

- `db_search_node`
  - 执行结构化查询
  - 当前主要支持“权限范围”“可访问文件”“最近文件”等问题

- `web_search_node`
  - 执行外部 Web 搜索
  - 默认需用户画像显式允许

- `finalize_node`
  - 将证据摘要、子问题结果、引用整合成最终回答

### 6.3 RAG 层

- `src/rag/rag_service.py`
  - 核心 RAG 服务
  - 同时负责文档入库 `ingestion()` 和查询 `query()`

- `src/rag/ingestion/loader.py`
  - 多格式文件加载器
  - PDF 场景支持文本提取不足时回退到 OCR

- `src/rag/ingestion/chunker.py`
  - 按文件类型采用不同切块策略

- `src/rag/retrieval/dense.py`
  - 向量检索
  - 支持元数据过滤

- `src/rag/retrieval/bm25.py`
  - 两种 BM25 实现：
    - `lite`：基于 MongoDB 已存文档内容做内存 BM25
    - `es`：基于 Elasticsearch

- `src/rag/retrieval/hybrid.py`
  - 将向量检索与 BM25 检索做 RRF 融合

- `src/rag/rerank/reranker.py`
  - 统一封装 Rerank
  - 可配置为 LLM 或 CrossEncoder

- `src/rag/context/builder.py`
  - 将候选文档去重、截断并拼成上下文字符串

- `src/tools/rag_tool.py`
  - 对复杂拆解问题支持 multi-pass RAG
  - 对多个子问题结果再做聚合

### 6.4 服务接口层

- `service/router/agent/chat.py`
  - SSE 流式聊天接口
  - 同步写入聊天消息与 run 报告

- `service/router/agent/query.py`
  - 非流式 Agent 查询接口

- `service/router/agent/admin_monitor.py`
  - 管理端运行监控接口

- `service/router/file/upload.py`
  - 文件上传与异步入库入口

- `service/router/users/login.py`
  - 登录接口，返回 JWT

- `service/router/users/profile.py`
  - 查询/更新用户画像

### 6.5 前端层

- `web_service/src/view/chat/Chat.vue`
  - 聊天主页面
  - 会管理当前会话、消息流、用户画像加载

- `web_service/src/components/ChatWindow.vue`
  - 展示消息、引用、trace、action_history

- `web_service/src/view/admin/*`
  - 管理端监控页
  - 包括概览、运行明细和可视化图表

## 7. 数据存储设计

### PostgreSQL

用于两类数据：

- SQLModel 业务表
  - `users`
  - `file`
  - `department`
  - `role`
  - `role_department`
  - `user_profile`
- PGVector 向量表
  - 存储文档切块后的 embedding

### MongoDB

主要用于：

- 文档切块原文存储
- 聊天会话 `chat_sessions`
- 聊天消息 `chat_messages`
- 运行记录 `chat_message_runs`
- QA/评估数据

### Elasticsearch

可选。

当 `BM25_RETRIEVAL_MODE=es` 时启用 Elasticsearch 作为 BM25 检索后端。注意当前代码的索引映射使用了：

- `ik_max_word`
- `ik_smart`

因此 Elasticsearch 需要提前安装 IK 分词器，否则索引创建会失败。

## 8. 运行前准备

### 基础环境

建议准备以下环境：

- Python `3.11`
- Node.js `18+`
- PostgreSQL，且已安装 `pgvector` 扩展
- MongoDB
- 可选：Elasticsearch + IK 分词器

### 关于旧文档说明

仓库早期 README 中出现过 Milvus 相关命令，但当前代码里的实际向量存储实现已经是：

- `PostgreSQL + PGVector`

因此现在以 PGVector 方案为准，Milvus 不是当前主链路的一部分。

## 9. 环境变量

项目通过根目录 `.env` 提供配置，具体字段以 `core/settings.py` 为准。下面给出最关键的一组配置示例：

```env
# PostgreSQL / PGVector
DATABASE_NAME=rag_agent
DATABASE_STRING=postgresql://user:password@127.0.0.1:5432/rag_agent
DATABASE_ASYNC_STRING=postgresql+asyncpg://user:password@127.0.0.1:5432/rag_agent
VECTOR_TABLE_NAME=rag_vector_store
EMBEDDING_DIM=1024

# MongoDB
MONGODB_URL=mongodb://127.0.0.1:27017
MONGODB_DB_NAME=rag_agent
DOC_COLLECTION_NAME=documents
QA_COLLECTION_NAME=qa_benchmark

# Elasticsearch（可选）
ELASTICSEARCH_URL=http://127.0.0.1:9200
BM25_RETRIEVAL_MODE=lite

# 模型
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TYPE=cross-encoder
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_URL=https://api.deepseek.com
ZHIPUAI_API_KEY=your-zhipu-key

# Agent / Retrieval
RETRIEVER_TOP_K=5
RERANKER_TOP_K=5
RETRIEVAL_MIN_SCORE=0.1
RERANKER_MIN_SCORE=0.1
AGENT_MAX_STEPS=10
AGENT_CHAT_HISTORY_LIMIT=8
AGENT_OUTPUT_LEVEL=standard

# 文档处理
OCR_LANG=ch
OCR_MIN_SCORE=0.5
TXT_CHUNK_SIZE=500
TXT_CHUNK_OVERLAP=50
PDF_CHUNK_SIZE=500
PDF_CHUNK_OVERLAP=50
EXCEL_CHUNK_SIZE=500
EXCEL_CHUNK_OVERLAP=50

# 运行控制
MAX_RETRIES=3
MAX_TIMEOUT=60
DELETE_FILE=false
```

### 关键配置说明

- `BM25_RETRIEVAL_MODE`
  - `lite`：无需 Elasticsearch，直接使用 MongoDB 中的文档内容构建 BM25
  - `es`：使用 Elasticsearch 做 BM25

- `RERANKER_TYPE`
  - `llm`
  - `cross-encoder`

- `AGENT_OUTPUT_LEVEL`
  - `concise`
  - `standard`
  - `detailed`

## 10. 安装与启动

### 10.1 安装 Python 依赖

Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 10.2 准备数据库

至少需要：

- PostgreSQL + pgvector
- MongoDB

如果你暂时不需要 Elasticsearch，请设置：

```env
BM25_RETRIEVAL_MODE=lite
```

### 10.3 启动后端

```powershell
python app.py
```

默认监听：

- `http://127.0.0.1:1016`

后端启动后会做两件事：

- 初始化 SQLModel 表结构
- 挂载静态资源目录 `service/public`

### 10.4 启动前端

```powershell
cd web_service
npm install
npm run dev
```

前端默认是 Vite 开发服务器，通常访问：

- `http://127.0.0.1:5173/#/chat`

前端请求会自动指向：

- `http://<当前主机>:1016`

## 11. 文档入库流程

上传文档后的主流程如下：

1. 文件通过 `/file/upload` 上传
2. 原文件写入 `service/public/uploads/`
3. 生成 `DocumentMetadata`
4. 调用 `rag_service.ingestion()`
5. 文件被 `loader.py` 解析
6. 内容被 `chunker.py` 切块
7. 切块 embedding 写入 PGVector
8. 原始块内容写入 MongoDB 或 Elasticsearch
9. 文件状态更新为已完成或失败

### 支持的文件类型

- `txt`
- `doc/docx`
- `md/markdown`
- `pdf`
- `xls/xlsx`
- `csv`
- `pptx`
- `json`
- `jpeg/png/jpg/bmp/webp/tiff/tif`

## 12. 查询流程

### 12.1 直答流程

适合：

- 概念解释
- 语言润色
- 翻译
- 不依赖企业知识库的简单问题

### 12.2 RAG 流程

典型步骤：

1. 构建查询上下文
2. 按用户权限加入过滤条件
3. Dense 检索
4. BM25 检索
5. RRF 融合
6. Rerank
7. 构建上下文
8. 让 LLM 评估证据是否充分
9. 如果足够，则进入 `finalize`

### 12.3 DB Search 流程

适合结构化业务问题，例如：

- 我能访问哪些部门
- 我最近可访问的文件有哪些
- 我上传过哪些文件
- 当前角色可访问哪些部门

### 12.4 Web Search 流程

适合：

- 时效性问题
- 外部新闻/市场/政策类问题

注意：

- 默认只有当用户画像 `allow_web_search=true` 时才会走外部搜索

## 13. 用户画像

用户画像存储在 `user_profile` 表中，主要字段包括：

- `answer_style`：回答详细程度
- `preferred_language`：偏好语言
- `preferred_topics`：偏好主题
- `prefers_citations`：是否偏好显示引用
- `allow_web_search`：是否允许联网搜索
- `profile_notes`：附加说明

它会影响：

- 最终回答长度
- 查询改写和扩展策略
- 是否注入偏好主题作为弱提示
- 是否允许外部搜索
- 是否在最终回答中带引用

## 14. API 概览

### 用户与画像

- `POST /user/login`
- `GET /user/profile`
- `PUT /user/profile`

### 文件

- `POST /file/upload`
- `GET /file/query_file`

### Agent

- `POST /agent/query`
- `POST /agent/chat/stream`
- `GET /agent/sessions`
- `GET /agent/sessions/{session_id}/messages`
- `DELETE /agent/sessions/{session_id}`

### 管理监控

- `GET /agent/admin/monitor/overview`
- `GET /agent/admin/monitor/runs`

## 15. 前端页面

### 聊天页

功能包括：

- 登录
- 会话列表
- 新建会话
- 删除会话
- 流式接收回答
- 查看引用
- 查看 trace / action history
- 编辑用户画像

### 管理页

功能包括：

- 今日请求量
- 活跃用户数
- 活跃会话数
- tokens 与估算成本
- 失败率
- Action 分布
- 模型分布
- 最近运行记录

## 16. 测试

当前仓库中较完整的自动化测试集中在 `preferred_topics` 相关逻辑，示例命令：

```bash
python -m unittest tests.test_preferred_topics
```

测试覆盖内容主要包括：

- 用户偏好主题提取
- 查询扩展与主题引导
- 运行报告中的主题使用摘要

## 17. 常见问题

### 17.1 为什么上传后检索不到内容？

优先检查：

- 文件状态是否已完成
- PostgreSQL 向量表是否成功写入
- MongoDB 文档块是否已写入
- 检索过滤条件是否把文档挡掉了
- `retrieval_min_score` / `reranker_min_score` 是否过高

### 17.2 为什么 Elasticsearch 启动了但入库报错？

当前索引映射使用 IK 分词器。如果你的 ES 没有安装 IK 插件，索引创建会失败。

### 17.3 为什么 Web Search 没生效？

先检查：

- 用户画像里的 `allow_web_search` 是否开启
- `ZHIPUAI_API_KEY` 是否可用
- 查询是否真的被策略识别为时效性外部问题

### 17.4 为什么登录后没有数据？

后端不会自动创建业务用户数据，`/user/login` 依赖数据库中已有用户记录。

## 18. 开发建议

如果你准备继续扩展这个项目，建议优先从以下位置入手：

- 改路由策略：`src/agent/policy.py`
- 改节点流程：`src/agent/graph.py`、`src/nodes/`
- 改提示词：`src/prompts/`
- 改检索策略：`src/rag/retrieval/`、`src/rag/rerank/`
- 改 API：`service/router/`
- 改前端交互：`web_service/src/`

## 19. 当前实现特点与限制

### 特点

- 项目主链路完整，已经具备“上传文档 -> 检索问答 -> 会话留痕 -> 监控分析”的闭环
- 权限过滤和用户画像已经接入 Agent 决策
- 管理端能查看运行明细，而不仅是聊天结果

### 限制

- README 之外的注释中存在少量中文编码乱码
- 自动化测试覆盖面还不够宽，RAG 主链路测试偏少
- 部分结构化查询能力仍是规则驱动，适用范围有限
- 当前登录逻辑依赖已有用户数据初始化

## 20. 入口文件速查

- 服务启动：`app.py`
- FastAPI 应用：`service/server.py`
- Agent 图入口：`src/agent/graph.py`
- Agent 运行入口：`src/agent/runner.py`
- RAG 主服务：`src/rag/rag_service.py`
- 前端主页面：`web_service/src/view/chat/Chat.vue`

---

如果你是第一次接手这个仓库，建议按下面顺序阅读源码：

1. `service/server.py`
2. `src/agent/graph.py`
3. `src/agent/policy.py`
4. `src/nodes/`
5. `src/rag/rag_service.py`
6. `service/router/agent/chat.py`
7. `web_service/src/view/chat/Chat.vue`

这样最容易把“接口层 -> Agent 编排层 -> RAG 执行层 -> 前端交互层”串起来。
