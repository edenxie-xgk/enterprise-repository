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
            logger.info(f"📤开始向量入库 等待加载文件个数：{settings.await_upload_file_num}")

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

    def query(self,query_context:RagContext,user_context:dict):
        """检索RAG内容"""



        # 输入规范化、重写
        # 1. Query处理
        # print("***" * 50)
        # print(f"💻对用户数据进行规范化、重写")
        # query_result = rewrite_tool.run(query)
        # print(query_result)

        search_queries = []
        if query_context.query and query_context.query.strip()!='':
            search_queries.append(query_context.query)
        if query_context.rewritten_query and query_context.rewritten_query.strip() != '':
            search_queries.append(query_context.rewritten_query)
        if query_context.expand_query:
            search_queries.extend([item for item in query_context.expand_query if item and item.strip() != ""])
        if query_context.decompose_query:
            search_queries.extend([item for item in query_context.decompose_query if item and item.strip() != ""])

        if not search_queries:
            return RAGResult(
                answer="暂无查询句子",
                documents=[],
                is_sufficient=False
            )

        # 2. 召回检索
        print("🎯hybrid检索")
        hybrid_docs = HybridRetriever(self.dense_retriever, self.bm25_retriever).run(search_queries)
        for doc in hybrid_docs[:3]:
            print(doc['content'])

        doc_ids = []
        docs= []
        for doc in hybrid_docs:
            if doc['node_id'] not in doc_ids: #去重
                doc_ids.append(doc['node_id'])
                docs.append(doc)

        # 3.reranker重排
        print("***" * 50)
        print("🥇reranker重排")
        docs = self.rerank.run(f"{query_context.query} {query_context.rewritten_query}", docs[:settings.retriever_top_k])
        for doc in docs[:3]:
            print(doc["rerank_score"], doc["content"])
            print("***" * 50)


        # 4. 拼接上下文
        print("***" * 50)
        print("📜去重、截断、拼接上下文")
        context_builder = ContextBuilder()
        context = context_builder.run(docs)
        print(context)

        # 5. 生成答案
        print("***" * 50)
        print("💬生成答案")
        response = generate_answer(self.chatgpt_llm,f"{query_context.query} {query_context.rewritten_query}", context)

        # documents = [node_id in [item['node_id'] for item in docs ]  for node_id in response.citations]

        # 6. 验证跟翻译
        verify = verify_answer(llm=self.llm, context=response.citations, answer=response.answer)
        if not verify:
            return RAGResult(
                answer=response.answer,
                documents=docs,
                is_sufficient=True
            )
        else:
            # return translate(llm=self.llm, query=query)
            return RAGResult(
                answer=response.answer,
                documents=docs,
                is_sufficient=False
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
                print(dense_docs)
                dense_docs = [item for item in dense_docs if item['score'] > 0.8]
                if dense_docs:
                    dense_docs = sorted(dense_docs, key=lambda x: x['score'], reverse=True)[
                        :min(3, settings.reranker_top_k - 2)]
                qa_list = generate_qa(llm=self.llm, nodes=[doc, *dense_docs], metadata=doc['metadata'])
                if qa_list:
                    self.qa_collection.insert_many(qa_list)
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
        print(f"🎯召回数据：{retrieval_report}")
        rerank_report = evaluate_rerank(self.dense_retriever,self.rerank,benchmark_data)
        print(f"🥇重排数据：{rerank_report}")
        generation_report = evaluate_generation(
            llm=self.llm,
            benchmark=benchmark_data,
            retriever=self.dense_retriever,
            rerank=self.rerank
        )
        print(f"💬答案数据：{rerank_report}")
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
    # rag_service.generation_evaluate_data()