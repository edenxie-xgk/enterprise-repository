import time

import os

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.database.es import ElasticsearchClient
from src.database.mongodb import mongodb_client
from src.models.embedding import embed_model
from src.models.llm import deepseek_llm, chatgpt_llm
from src.rag.context.builder import ContextBuilder
from src.rag.evaluate.generation import evaluate_generation
from src.rag.evaluate.qa import generate_qa
from src.rag.evaluate.rerank import evaluate_rerank
from src.rag.evaluate.retrieval import evaluate_retrieval
from src.rag.generation.answer_verify import verify_answer
from src.rag.generation.generator import  generate_answer
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
        is_sufficient: bool,
        fail_reason=None,
        retrieval_queries=None,
        diagnostics=None,
        success: bool = True,
    ) -> RAGResult:
        return RAGResult(
            answer=answer,
            documents=documents or [],
            citations=citations or [],
            is_sufficient=is_sufficient,
            fail_reason=fail_reason,
            success=success,
            tool_name="rag",
            retrieval_queries=retrieval_queries or [],
            diagnostics=diagnostics or [],
        )

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

            elapsed_time = time.time() - start_time
            logger.info(f"[rag向量存储成功]:存储文件:${metadata.file_path} 用时:${elapsed_time}s")
            return True
        except Exception as e:
            logger.error(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}")
            raise Exception(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}") from e

    def query(self,query_context:RagContext,user_context:dict, previous_result: RAGResult | None = None):
        """检索RAG内容"""
        search_queries = []
        if query_context.query and query_context.query.strip()!='':
            search_queries.append(query_context.query)
        if query_context.rewritten_query and query_context.rewritten_query.strip() != '':
            search_queries.append(query_context.rewritten_query)
        if query_context.expand_query:
            search_queries.extend([item for item in query_context.expand_query if item and item.strip() != ""])
        if query_context.decompose_query:
            search_queries.extend([item for item in query_context.decompose_query if item and item.strip() != ""])
        search_queries = self._dedupe_queries(search_queries)

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
            if query_context.use_retrieval or not previous_result or not previous_result.documents:
                print("hybrid retrieval")
                hybrid_docs = HybridRetriever(self.dense_retriever, self.bm25_retriever).run(
                    search_queries,
                    top_k=query_context.retrieval_top_k,
                    filters=query_context.filters,
                )
                doc_ids = []
                for doc in hybrid_docs:
                    if doc['node_id'] not in doc_ids:
                        doc_ids.append(doc['node_id'])
                        docs.append(doc)
                retrieval_diagnostics = ["hybrid_retrieval_executed"]
            else:
                docs = self._normalize_candidate_docs(previous_result.documents)
                retrieval_diagnostics = ["retrieval_reused_previous_docs"]

            if not docs:
                return self._build_rag_result(
                    "未检索到相关文档",
                    is_sufficient=False,
                    fail_reason="no_data",
                    retrieval_queries=search_queries,
                    diagnostics=retrieval_diagnostics + ["hybrid_retrieval_returned_no_docs"],
                )

            if query_context.use_rerank:
                print("***" * 50)
                print("reranker")
                docs = self.rerank.run(
                    f"{query_context.query} {query_context.rewritten_query}",
                    docs[:query_context.retrieval_top_k],
                    top_k=query_context.rerank_top_k,
                )
                rerank_diagnostics = ["reranker_executed"]
            else:
                docs = docs[:query_context.rerank_top_k]
                rerank_diagnostics = ["reranker_skipped"]

            if not docs:
                return self._build_rag_result(
                    "检索到了候选文档，但重排后没有足够相关的结果",
                    is_sufficient=False,
                    fail_reason="bad_ranking",
                    retrieval_queries=search_queries,
                    diagnostics=retrieval_diagnostics + rerank_diagnostics + ["reranker_filtered_all_docs"],
                )

            print("***" * 50)
            print("build context")
            context_builder = ContextBuilder()
            context = context_builder.run(docs)
            print(context)

            print("***" * 50)
            print("generate answer")
            response = generate_answer(
                self.chatgpt_llm,
                f"{query_context.query} {query_context.rewritten_query}",
                context,
            )

            if response.is_sufficient:
                verify = verify_answer(llm=self.chatgpt_llm, context=context, answer=response.answer)
                if verify:
                    return self._build_rag_result(
                        response.answer,
                        documents=docs,
                        citations=response.citations,
                        is_sufficient=True,
                        retrieval_queries=search_queries,
                        diagnostics=retrieval_diagnostics + rerank_diagnostics + ["generation_sufficient", "answer_verified"],
                    )

                return self._build_rag_result(
                    response.answer,
                    documents=docs,
                    citations=response.citations,
                    is_sufficient=False,
                    fail_reason="verification_failed",
                    retrieval_queries=search_queries,
                    diagnostics=retrieval_diagnostics + rerank_diagnostics + ["generation_sufficient", "answer_verification_failed"],
                )

            return self._build_rag_result(
                response.answer,
                documents=docs,
                citations=response.citations,
                is_sufficient=False,
                fail_reason=response.fail_reason,
                retrieval_queries=search_queries,
                diagnostics=retrieval_diagnostics + rerank_diagnostics + ["generation_insufficient"],
            )
        except Exception as exc:
            logger.exception("RAG query failed")
            return self._build_rag_result(
                "RAG查询失败",
                is_sufficient=False,
                fail_reason="tool_error",
                retrieval_queries=search_queries,
                diagnostics=["rag_query_exception", str(exc)],
                success=False,
            )


    def generation_evaluate_data(self):
        docs = self.doc_collection.find({"state":2}).to_list()
        if not docs:
            return {
                "message": "暂无数据生成评估数据",
                "success": True
            }

        error_nodes = []
        for doc in docs:
            try:
                dense_docs = self.dense_retriever.run([doc['content']], top_k=5)
                dense_docs = [item for item in dense_docs if item['dense_score'] > 0.8]
                if dense_docs:
                    dense_docs = sorted(dense_docs, key=lambda x: x['dense_score'], reverse=True)[
                        :min(3, settings.reranker_top_k - 2)]
                qa_list = generate_qa(llm=self.llm, nodes=[doc, *dense_docs], metadata=doc['metadata'])
                if qa_list:
                    self.qa_collection.insert_many(qa_list)
                self.doc_collection.update_one({"node_id":doc['node_id']},{"$set":{"state":1}})
            except Exception:
                error_nodes.append(doc['node_id'])
        if error_nodes:
            return {
                "message":"生成评估数据成功",
                "success":True
            }
        else:
            return {
                "error_nodes":error_nodes,
                "success": False
            }

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
