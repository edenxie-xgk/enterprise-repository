import time

import os

from tqdm import tqdm

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.database.es import ElasticsearchClient
from src.database.mongodb import mongodb_client
from src.graph import graph_service
from src.models.embedding import embed_model
from src.models.llm import deepseek_llm, chatgpt_llm
from src.rag.context.builder import ContextBuilder
from src.rag.evaluate.generation import evaluate_generation
from src.rag.evaluate.qa import generate_qa
from src.rag.evaluate.rerank import evaluate_rerank
from src.rag.evaluate.retrieval import evaluate_retrieval
from src.rag.generation.generator import evaluate_evidence
from src.rag.rerank.reranker import Reranker
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.hybrid import HybridRetriever
from src.types.rag_type import RAGResult, RagContext
from utils.logger_handler import logger
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from src.rag.ingestion.loader import  load_file
from src.rag.store.vector_store import vector_store
from src.rag.ingestion.chunker import chunk_file
from src.rag.retrieval.bm25 import BM25LiteRetriever,ESRetriever


class RAGService:
    def __init__(self):
        self.llm = deepseek_llm
        self.chatgpt_llm = chatgpt_llm
        Settings.embed_model = embed_model
        Settings.llm = self.llm
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        self.dense_retriever = DenseRetriever(vector_store=vector_store, storage_context=self.storage_context)

        if  settings.bm25_retrieval_mode == "lite":
            self.doc_collection = mongodb_client.get_collection(settings.doc_collection_name)
        elif settings.bm25_retrieval_mode == "es":
            self.doc_collection = ElasticsearchClient(settings.doc_collection_name)
        else:
            raise Exception('bm25_retrieval_mode 参数错误')

        self.bm25_retriever = None
        self._create_bm25_retrieval()

        self.rerank = Reranker(llm=self.llm)
        self.update_doc_time = time.time()

        self.qa_collection = mongodb_client.get_collection(settings.qa_collection_name)

    @staticmethod
    def _cursor_to_list(cursor_or_rows):
        if cursor_or_rows is None:
            return []
        if hasattr(cursor_or_rows, "to_list"):
            return cursor_or_rows.to_list()
        return list(cursor_or_rows)

    @staticmethod
    def _matches_metadata_value(actual, expected) -> bool:
        if isinstance(expected, (list, tuple, set)):
            return any(RAGService._matches_metadata_value(actual, item) for item in expected)
        if actual == expected:
            return True
        return str(actual) == str(expected)

    @staticmethod
    def _dedupe_queries(queries: list[str]) -> list[str]:
        seen = set()
        result = []
        for query in queries:
            normalized = query.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _build_rag_result(
        answer: str,
        *,
        documents=None,
        citations=None,
        evidence_summary: str | None = None,
        is_sufficient: bool,
        fail_reason=None,
        retrieval_queries=None,
        retrieval_candidate_node_ids=None,
        rerank_node_ids=None,
        diagnostics=None,
        success: bool = True,
    ) -> RAGResult:
        return RAGResult(
            answer=answer,
            documents=documents or [],
            citations=citations or [],
            evidence_summary=evidence_summary if evidence_summary is not None else answer,
            is_sufficient=is_sufficient,
            fail_reason=fail_reason,
            success=success,
            name="rag",
            retrieval_queries=retrieval_queries or [],
            retrieval_candidate_node_ids=retrieval_candidate_node_ids or [],
            rerank_node_ids=rerank_node_ids or [],
            diagnostics=diagnostics or [],
        )

    @staticmethod
    def _extract_node_ids(documents) -> list[str]:
        """
        从文档集合中提取唯一的节点ID列表

        该方法处理多种格式的文档对象，提取其中的node_id字段：
        1. 支持对象属性方式获取（如Pydantic模型对象）
        2. 支持字典方式获取
        3. 自动去重，确保返回的ID列表唯一

        Args:
            documents: 待处理的文档集合，可以是None、列表或可迭代对象，
                      包含字典或有node_id属性的对象

        Returns:
            list[str]: 去重后的节点ID列表，按首次出现的顺序排列
        """
        node_ids = []  # 存储结果的有序列表
        seen = set()   # 用于快速查找已处理的节点ID
        for doc in documents or []:  # 处理None输入情况
            # 尝试从不同格式的文档中获取node_id
            if hasattr(doc, "node_id"):
                node_id = getattr(doc, "node_id", None)  # 从对象属性获取
            elif isinstance(doc, dict):
                node_id = doc.get("node_id")  # 从字典键获取
            else:
                node_id = None  # 不支持的格式

            # 跳过无效或重复的node_id
            if not node_id or node_id in seen:
                continue

            seen.add(node_id)  # 记录已处理的ID
            node_ids.append(node_id)  # 添加到结果列表
        return node_ids

    @staticmethod
    def _normalize_candidate_docs(documents):
        """
        规范化候选文档格式，统一转换为字典形式

        该方法用于处理不同格式的文档对象，将其统一转换为字典形式：
        1. 如果文档对象有 model_dump() 方法，调用该方法转换为字典
        2. 如果已经是字典形式，直接保留
        3. 其他格式将被忽略

        Args:
            documents: 待处理的文档列表，可以是包含不同格式文档的列表或None

        Returns:
            list: 规范化后的文档字典列表
        """
        normalized = []
        # 遍历文档列表（如果输入为None则视为空列表处理）
        for doc in documents or []:
            # 检查文档是否有 model_dump 方法（如Pydantic模型对象）
            if hasattr(doc, "model_dump"):
                normalized.append(doc.model_dump())
            # 检查是否是字典类型
            elif isinstance(doc, dict):
                normalized.append(doc)
        return normalized

    def _list_docs_for_qa_generation(
        self,
        *,
        source_state: int = 2,
        limit: int | None = None,
        department_id=None,
    ) -> list[dict]:
        if not hasattr(self.doc_collection, "find"):
            raise RuntimeError("当前文档存储后端不支持 QA 数据脚本化生成，请切换到支持 find 的实现。")

        docs = self._cursor_to_list(self.doc_collection.find({"state": source_state}))

        if department_id is not None:
            docs = [
                doc
                for doc in docs
                if self._matches_metadata_value((doc.get("metadata") or {}).get("department_id"), department_id)
            ]

        if limit is not None:
            docs = docs[: max(limit, 0)]

        return docs

    def generate_qa_dataset(
        self,
        *,
        source_state: int = 2,
        mark_source_state: int | None = 1,
        limit: int | None = None,
        department_id=None,
        dense_score_threshold: float = 0.8,
        dense_top_k: int = 5,
        max_related_docs: int = 3,
        dry_run: bool = False,
    ) -> dict:
        docs = self._list_docs_for_qa_generation(
            source_state=source_state,
            limit=limit,
            department_id=department_id,
        )
        if not docs:
            return {
                "message": "暂无数据生成评估数据",
                "success": True,
                "processed_docs": 0,
                "generated_qa_count": 0,
                "inserted_qa_count": 0,
                "updated_doc_count": 0,
                "error_nodes": [],
                "generated_rows": [],
                "dry_run": dry_run,
            }

        if mark_source_state is not None and not hasattr(self.doc_collection, "update_one"):
            raise RuntimeError("当前文档存储后端不支持更新文档状态，无法标记 QA 生成结果。")

        generated_rows = []
        error_nodes = []
        empty_qa_nodes = []
        inserted_qa_count = 0
        updated_doc_count = 0

        for doc in tqdm(docs, desc="生成评估数据"):
            node_id = doc.get("node_id")
            metadata = doc.get("metadata") or {}
            try:
                retrieval_filters = {}
                if metadata.get("department_id") is not None:
                    retrieval_filters["department_id"] = metadata.get("department_id")

                dense_docs = self.dense_retriever.run(
                    [doc.get("content") or ""],
                    top_k=max(1, dense_top_k),
                    filters=retrieval_filters or None,
                )
                dense_docs = [
                    item
                    for item in dense_docs
                    if item.get("node_id") != node_id and (item.get("dense_score") or 0.0) >= dense_score_threshold
                ]
                dense_docs = sorted(dense_docs, key=lambda item: item.get("dense_score") or 0.0, reverse=True)[
                    : max(max_related_docs, 0)
                ]

                qa_list = generate_qa(llm=self.llm, nodes=[doc, *dense_docs], metadata=metadata)
                if not qa_list:
                    empty_qa_nodes.append(
                        {
                            "node_id": node_id,
                            "reason": "no_qa_generated",
                        }
                    )
                    continue

                generated_rows.extend(qa_list)
                if not dry_run:
                    self.qa_collection.insert_many(qa_list)
                    inserted_qa_count += len(qa_list)

                if not dry_run and mark_source_state is not None:
                    self.doc_collection.update_one({"node_id": node_id}, {"$set": {"state": mark_source_state}})
                    updated_doc_count += 1
            except Exception as exc:
                logger.exception("生成 QA 数据失败: node_id=%s", node_id)
                error_nodes.append(
                    {
                        "node_id": node_id,
                        "error": str(exc),
                    }
                )

        success = bool(generated_rows) and not error_nodes and not empty_qa_nodes
        if success:
            message = "生成评估数据成功"
        elif error_nodes:
            message = "部分评估数据生成失败"
        elif empty_qa_nodes:
            message = "未生成任何有效 QA 数据"
        else:
            message = "未生成评估数据"

        return {
            "message": message,
            "success": success,
            "processed_docs": len(docs),
            "generated_qa_count": len(generated_rows),
            "inserted_qa_count": inserted_qa_count,
            "updated_doc_count": updated_doc_count,
            "error_nodes": error_nodes,
            "empty_qa_count": len(empty_qa_nodes),
            "empty_qa_nodes": empty_qa_nodes,
            "generated_rows": generated_rows,
            "dry_run": dry_run,
        }

    @staticmethod
    def _is_complex_evidence_query(query_context: RagContext) -> bool:
        text = " ".join(
            [
                query_context.query or "",
                query_context.rewritten_query or "",
                " ".join(query_context.decompose_query or []),
            ]
        ).lower()
        markers = [
            "compare",
            "comparison",
            "difference",
            "differences",
            "commonality",
            "commonalities",
            "analysis",
            "risk",
            "different meetings",
            "different speakers",
            "共同点",
            "差异",
            "区别",
            "比较",
            "对比",
            "分析",
            "风险",
        ]
        return len(query_context.decompose_query or []) >= 2 or any(marker in text for marker in markers)

    @staticmethod
    def _merge_fallback_docs(primary_docs, fallback_docs, keep_count: int):
        """合并主文档和备用文档，并确保结果不重复且不超过指定数量

        该方法用于在重排序阶段合并主文档集和备用文档集，确保：
        1. 合并后的文档不重复（基于node_id去重）
        2. 合并后的文档数量不超过keep_count指定的数量
        3. 优先保留主文档集中的文档

        Args:
            primary_docs: 主文档列表，优先保留的文档集
            fallback_docs: 备用文档列表，在主文档不足时补充
            keep_count: 最终返回的文档最大数量限制

        Returns:
            list: 合并后的文档列表，已去重且不超过keep_count限制
        """
        merged = []
        seen = set()
        for doc in list(primary_docs or []) + list(fallback_docs or []):
            node_id = doc.get("node_id") if isinstance(doc, dict) else getattr(doc, "node_id", None)
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            merged.append(doc)
            if len(merged) >= keep_count:
                break
        return merged

    @staticmethod
    def _looks_self_declared_insufficient(text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return True
        markers = [
            "insufficient evidence",
            "not enough evidence",
            "unable to answer",
            "cannot answer",
            "no relevant evidence",
            "缺乏",
            "不足",
            "无法回答",
            "无法稳定回答",
            "没有足够",
            "未检索到",
        ]
        return any(marker in normalized for marker in markers)

    def _apply_evidence_guardrails(self, query_context: RagContext, docs, response):
        """应用证据护栏检查，评估RAG响应是否充分可靠

        该方法对RAG生成的响应进行多维度验证，确保结果质量：
        1. 检查证据摘要是否为空
        2. 检查是否有引用来源
        3. 验证来源数量是否满足最低要求
        4. 检测响应是否自声明不足

        Args:
            query_context: 包含查询上下文的RagContext对象
            docs: 检索到的文档列表
            response: RAG生成的响应对象，包含is_sufficient、fail_reason等属性

        Returns:
            tuple: 包含三个元素的元组：
                - is_sufficient (bool): 证据是否充分
                - fail_reason (str): 失败原因（如不足）
                - diagnostics (list): 诊断信息列表
        """
        # 初始化诊断信息和基础变量
        diagnostics = []
        is_sufficient = bool(response.is_sufficient)
        fail_reason = response.fail_reason
        evidence_summary = (response.evidence_summary or "").strip()
        citations = list(response.citations or [])

        # 获取文档的唯一节点ID并计算最小所需来源数
        unique_node_ids = self._extract_node_ids(docs)
        # 复杂查询需要至少2个来源，普通查询1个
        min_required_sources = 2 if self._is_complex_evidence_query(query_context) else 1

        # 检查1：证据摘要是否为空
        if not evidence_summary:
            is_sufficient = False
            fail_reason = fail_reason or "insufficient_context"
            diagnostics.append("evidence_guard_empty_summary")

        # 检查2：是否有引用来源
        if not citations:
            is_sufficient = False
            fail_reason = fail_reason or "insufficient_context"
            diagnostics.append("evidence_guard_missing_citations")

        # 检查3：来源数量是否足够
        if len(unique_node_ids) < min_required_sources:
            is_sufficient = False
            fail_reason = fail_reason or "insufficient_context"
            diagnostics.append(f"evidence_guard_min_sources={min_required_sources}")

        # 检查4：响应是否自声明不足
        if self._looks_self_declared_insufficient(evidence_summary):
            is_sufficient = False
            fail_reason = fail_reason or "insufficient_context"
            diagnostics.append("evidence_guard_self_declared_insufficient")

        # 默认失败原因设置
        if not is_sufficient and not fail_reason:
            fail_reason = "insufficient_context"

        return is_sufficient, fail_reason, diagnostics

    def _create_bm25_retrieval(self):
        if settings.bm25_retrieval_mode == "lite":
            docs = self.doc_collection.find({}).to_list()
            if docs:
                self.bm25_retriever = BM25LiteRetriever(documents=docs)
                self.update_doc_time = time.time()
                settings.is_need_doc = False
        elif settings.bm25_retrieval_mode == "es":
            self.bm25_retriever = ESRetriever(es_client=self.doc_collection)
            self.update_doc_time = time.time()
            settings.is_need_doc = False
        else:
            raise Exception('bm25_retrieval_mode 参数错误')

    def ingestion(self, path:str, metadata: DocumentMetadata):
        """
        对文件路径的单个文件进行数据向量存储
        :param path: 文件路径
        :param metadata: 存储检索
        :return:
        """
        if not path or os.path.isdir(path):
            return False
        try:
            start_time = time.time()
            docs = load_file(path, metadata)
            logger.info(f"开始向量入库，等待加载文件个数：{settings.await_upload_file_num}")

            if not docs or len(docs) == 0:
                logger.error(f"[rag向量存储失败]:内容为空")
                return False

            nodes = chunk_file(docs)

            nodelist = []

            for chunk_index,node in enumerate(nodes):
                node.metadata['chunk_index'] = chunk_index+1
                nodes[chunk_index].metadata['chunk_index'] = chunk_index+1
                nodelist.append({"content":node.text, "metadata":node.metadata,'node_id':node.id_,"state":2}) #0代表删除数据 1代表已生成评估数据 2代表未生成评估数据

            # 向量入库
            VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=embed_model,
                show_progress=True,
            )

            # 文档入库
            self.doc_collection.insert_many(nodelist)

            # 财报事实图谱入库采用“旁路增强”方式：
            # 即使图谱抽取失败，也不阻断主 RAG 的文档入库。
            if settings.graph_enabled:
                try:
                    graph_result = graph_service.ingest_nodes(nodes)
                    logger.info(
                        f"[graph构建完成]: entities={graph_result.entity_count}, facts={graph_result.fact_count}"
                    )
                except Exception as graph_exc:
                    logger.warning(f"[graph构建失败但主链路继续]: {graph_exc}")

            elapsed_time = time.time() - start_time
            logger.info(f"[rag向量存储成功]:存储文件:${metadata.file_path} 用时:${elapsed_time}s")
            return True
        except Exception as e:
            logger.error(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}")
            raise Exception(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}") from e

    def query(self, query_context: RagContext, user_context: dict, previous_result: RAGResult | None = None):
        """执行RAG查询流程

        Args:
            query_context: 包含查询相关信息的上下文对象
            user_context: 用户上下文信息(当前未使用)
            previous_result: 前一次的RAG查询结果，可选

        Returns:
            RAGResult: 包含查询结果、文档、引用等信息的封装对象
        """
        # 构建搜索查询列表，合并原始查询、改写查询、扩展查询和分解查询
        search_queries = []
        if query_context.query and query_context.query.strip() != '':
            search_queries.append(query_context.query)
        if query_context.rewritten_query and query_context.rewritten_query.strip() != '':
            search_queries.append(query_context.rewritten_query)
        if query_context.expand_query:
            search_queries.extend([item for item in query_context.expand_query if item and item.strip() != ""])
        if query_context.decompose_query:
            search_queries.extend([item for item in query_context.decompose_query if item and item.strip() != ""])
        # 去重处理
        search_queries = self._dedupe_queries(search_queries)

        # 无有效查询时的返回处理
        if not search_queries:
            return self._build_rag_result(
                "暂无查询语句",
                is_sufficient=False,
                fail_reason="no_data",
                retrieval_queries=[],
                diagnostics=["empty_search_queries"],
            )

        try:
            docs = []
            retrieval_candidate_node_ids = []
            rerank_node_ids = []

            # 检索阶段：使用混合检索或复用之前的结果
            if query_context.use_retrieval or not previous_result or not previous_result.documents:
                print("hybrid retrieval")
                # 执行混合检索（稠密检索+BM25）
                hybrid_docs = HybridRetriever(self.dense_retriever, self.bm25_retriever).run(
                    search_queries,
                    top_k=query_context.retrieval_top_k,
                    filters=query_context.filters,
                )
                # 去重处理
                doc_ids = []
                for doc in hybrid_docs:
                    if doc['node_id'] not in doc_ids:
                        doc_ids.append(doc['node_id'])
                        docs.append(doc)
                retrieval_candidate_node_ids = self._extract_node_ids(docs)
                retrieval_diagnostics = ["hybrid_retrieval_executed"]
            else:
                # 复用之前的检索结果
                docs = self._normalize_candidate_docs(previous_result.documents)
                retrieval_candidate_node_ids = list(
                    getattr(previous_result, "retrieval_candidate_node_ids", []) or self._extract_node_ids(docs)
                )
                retrieval_diagnostics = ["retrieval_reused_previous_docs"]

            # 无检索结果处理
            if not docs:
                return self._build_rag_result(
                    "未检索到相关文档",
                    is_sufficient=False,
                    fail_reason="no_data",
                    retrieval_queries=search_queries,
                    retrieval_candidate_node_ids=retrieval_candidate_node_ids,
                    rerank_node_ids=rerank_node_ids,
                    diagnostics=retrieval_diagnostics + ["hybrid_retrieval_returned_no_docs"],
                )

            # 重排序阶段
            retrieval_docs = list(docs)
            fallback_keep_count = min(
                len(retrieval_docs),
                max(
                    query_context.rerank_top_k,
                    3 if self._is_complex_evidence_query(query_context) else 1,
                ),
            )
            if query_context.use_rerank:
                print("***" * 50)
                print("reranker")
                reranked_docs = self.rerank.run(
                    f"{query_context.query} {query_context.rewritten_query}",
                    retrieval_docs[:query_context.retrieval_top_k],
                    top_k=query_context.rerank_top_k,
                )
                rerank_diagnostics = ["reranker_executed", f"reranker_result_count={len(reranked_docs)}"]
                # 重排结果不足处理
                if not reranked_docs:
                    docs = retrieval_docs[:fallback_keep_count]
                    rerank_diagnostics.extend([
                        "reranker_fallback_applied",
                        "reranker_filtered_all_docs",
                        f"reranker_fallback_count={len(docs)}",
                    ])
                elif len(reranked_docs) < fallback_keep_count:
                    docs = self._merge_fallback_docs(reranked_docs, retrieval_docs, fallback_keep_count)
                    rerank_diagnostics.extend([
                        "reranker_fallback_supplemented",
                        f"reranker_fallback_count={len(docs)}",
                    ])
                else:
                    docs = reranked_docs

                rerank_node_ids = self._extract_node_ids(docs)
            else:
                docs = retrieval_docs[:query_context.rerank_top_k]
                rerank_node_ids = self._extract_node_ids(docs)
                rerank_diagnostics = ["reranker_skipped"]

            if not docs:
                return self._build_rag_result(
                    "检索到了候选文档，但重排后没有足够相关的结果",
                    is_sufficient=False,
                    fail_reason="bad_ranking",
                    retrieval_queries=search_queries,
                    retrieval_candidate_node_ids=retrieval_candidate_node_ids,
                    rerank_node_ids=rerank_node_ids,
                    diagnostics=retrieval_diagnostics + rerank_diagnostics + ["reranker_fallback_failed"],
                )

            print("***" * 50)
            print("build context")
            context_builder = ContextBuilder()
            context = context_builder.run(docs)
            print(context)

            # 证据评估阶段
            print("***" * 50)
            print("evaluate evidence")
            response = evaluate_evidence(
                self.chatgpt_llm,
                f"{query_context.query} {query_context.rewritten_query}",
                context,
            )
            guarded_is_sufficient, guarded_fail_reason, evidence_guard_diagnostics = self._apply_evidence_guardrails(
                query_context,
                docs,
                response,
            )

            if guarded_is_sufficient:
                return self._build_rag_result(
                    response.evidence_summary,
                    documents=docs,
                    citations=response.citations,
                    evidence_summary=response.evidence_summary,
                    is_sufficient=True,
                    retrieval_queries=search_queries,
                    retrieval_candidate_node_ids=retrieval_candidate_node_ids,
                    rerank_node_ids=rerank_node_ids,
                    diagnostics=retrieval_diagnostics + rerank_diagnostics + evidence_guard_diagnostics + ["evidence_sufficient"],
                )

            return self._build_rag_result(
                response.evidence_summary,
                documents=docs,
                citations=response.citations,
                evidence_summary=response.evidence_summary,
                is_sufficient=False,
                fail_reason=guarded_fail_reason,
                retrieval_queries=search_queries,
                retrieval_candidate_node_ids=retrieval_candidate_node_ids,
                rerank_node_ids=rerank_node_ids,
                diagnostics=retrieval_diagnostics + rerank_diagnostics + evidence_guard_diagnostics + ["evidence_insufficient"],
            )

        except Exception as exc:
            # 异常处理
            logger.exception("RAG query failed")
            return self._build_rag_result(
                "RAG查询失败",
                is_sufficient=False,
                fail_reason="tool_error",
                retrieval_queries=search_queries,
                diagnostics=["rag_query_exception", str(exc)],
                retrieval_candidate_node_ids=[],
                rerank_node_ids=[],
                success=False,
            )


    def generation_evaluate_data(self):
        summary = self.generate_qa_dataset(
            source_state=2,
            mark_source_state=1,
            dense_score_threshold=0.8,
            dense_top_k=5,
            max_related_docs=3,
        )
        summary.pop("generated_rows", None)
        return summary

    # 评估
    def benchmark(self):
        benchmark_data = self.qa_collection.find({'state':0}).to_list()

        if not benchmark_data:
            return {
                "message":"暂无新评估数据",
                "success": True
            }

        retrieval_report = evaluate_retrieval(self.dense_retriever,benchmark_data)
        print(f"retrieval report: {retrieval_report}")
        rerank_report = evaluate_rerank(self.dense_retriever,self.rerank,benchmark_data)
        print(f"rerank report: {rerank_report}")
        generation_report = evaluate_generation(
            llm=self.llm,
            benchmark=benchmark_data,
            retriever=self.dense_retriever,
            rerank=self.rerank
        )
        print(f"generation report: {rerank_report}")
        return {
            "retrieval_report":retrieval_report,
            "rerank_report":rerank_report,
            "generation_report":generation_report
        }


rag_service = RAGService()

if  __name__ == "__main__":
    # data = DocumentMetadata(
    #      department_id=1,
    #      department_name="TQ",
    #      user_id=1,
    #      user_name="EdenXie",
    #      file_path="public\\uploads\\TQ\\a1.png",
    #      file_name= "a1.png",
    #      file_size=100,
    #      file_type="png",
    #      source="png"
    #  )
    # rag_service.ingestion("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\a1.png",data)
    data = DocumentMetadata(
        department_id=1,
        department_name="TQ",
        user_id=1,
        user_name="EdenXie",
        file_path="public\\uploads\\OE\\fund_report_1.pdf",
        file_name="fund_report_1.pdf",
        file_size=100,
        file_type="pdf",
        source="pdf"
    )
    # rag_service.ingestion("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\文档上传测试.pdf", data)
    # rag_service.ingestion("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\OE\\fund_report_1.pdf",data)
    # rag_service.benchmark()

    rag_service.generation_evaluate_data()
