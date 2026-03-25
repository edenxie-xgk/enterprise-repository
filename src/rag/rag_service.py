import time

import os

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.database.es import ElasticsearchClient
from src.database.mongodb import mongodb_client
from src.models.embedding import embed_model
from src.models.llm import deepseek_llm, chatgpt_llm
from src.rag.context.builder import ContextBuilder
from src.rag.evaluate.qa import generate_qa
from src.rag.generation.answer_verify import verify_answer
from src.rag.generation.generator import  generate_answer
from src.rag.generation.translate import translate
from src.rag.query.query_processor import QueryProcessor
from src.rag.rerank.reranker import Reranker
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.hybrid import HybridRetriever
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
            if not docs or len(docs) == 0:
                logger.error(f"[rag向量存储失败]:内容为空")
                return False

            nodes = chunk_file(docs)



            nodelist = []
            qa_list = []
            for node in nodes:
                nodelist.append({"content":node.text, "metadata":node.metadata,'node_id':node.id_})
                qa_data = generate_qa(llm=self.llm,chunk=node.text,node_id=node.id_,metadata=node.metadata)
                qa_list.extend(qa_data)

            # 向量入库
            VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=embed_model,
                show_progress=True,
            )

            # qa入库
            self.qa_collection.insert_many(qa_list)

            # 文档入库
            self.doc_collection.insert_many(nodelist)



            elapsed_time = time.time() - start_time
            logger.info(f"[rag向量存储成功]:存储文件:${metadata.file_path} 用时:${elapsed_time}s")
            return True
        except Exception as e:
            logger.error(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}")
            raise Exception(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}") from e

    def query(self,query:str,user_context:dict):
        """检索RAG内容"""

        # 输入规范化、重写
        # 1. Query处理
        print("***" * 50)
        print(f"💻对用户数据进行规范化、重写")
        query_processor = QueryProcessor(llm=self.llm)
        query_result = query_processor.run(query)
        print(query_result)

        # 2. Dense检索
        print("***" * 50)
        print(f"⌛检索数据")
        print(f"🎯Dense检索")
        docs = []

        dense_docs = self.dense_retriever.run(query_result.search_queries)
        docs.extend(dense_docs)
        for doc in dense_docs[:3]:
            print(doc["score"], doc["content"])
            print("***" * 50)


        if not self.bm25_retriever or (time.time() - self.update_doc_time > settings.update_doc_time and settings.is_need_doc):  #定期更新文档
            self._create_bm25_retrieval()
        if self.bm25_retriever:
            print("🎯bm25检索")
            bm25_docs = self.bm25_retriever.run(query_result.search_queries)
            docs.extend(bm25_docs)
            for doc in bm25_docs[:3]:
                print(doc["bm25_score"], doc["content"])
                print("***" * 50)


            print("🎯hybrid检索")
            hybrid_docs = HybridRetriever(self.dense_retriever, self.bm25_retriever).run(query_result.search_queries)
            docs.extend(hybrid_docs)
            for doc in hybrid_docs[:3]:
                print(doc['content'])

        # 3.reranker重排
        print("***" * 50)
        print("🥇reranker重排")
        rerank = Reranker(llm=self.llm)
        docs = rerank.run(query_result.rewrite_query, docs[:3])
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
        response = generate_answer(query_result.rewrite_query, context)
        print(response.answer)

        # 6. 验证跟翻译
        verify = verify_answer(llm=self.llm, context=response.citations, answer=response.answer)
        if not verify:
            return response.answer
        else:
            return translate(llm=self.llm, query=query)


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
        file_path="public\\uploads\\TQ\\文档上传测试.pdf",
        file_name="文档上传测试.pdf",
        file_size=100,
        file_type="pdf",
        source="pdf"
    )
    # rag_service.ingestion("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\文档上传测试.pdf", data)
    rag_service.ingestion("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\文档上传测试.pdf",data)
    # print(rag_service.query("Give me a brief introduction to financial knowledge",{}))
