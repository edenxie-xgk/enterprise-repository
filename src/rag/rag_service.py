import time

import os

from core.custom_types import DocumentMetadata
from src.models.embedding import embed_model
from src.models.llm import deepseek_llm
from src.rag.context.builder import ContextBuilder
from src.rag.generation.generator import Generator
from src.rag.query.query_processor import QueryProcessor
from src.rag.rerank.reranker import Reranker
from src.rag.retrieval.dense import DenseRetriever
from utils.logger_handler import logger
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from src.rag.ingestion.loader import  load_file
from src.rag.store.vector_store import vector_store
from src.rag.ingestion.chunker import chunk_file


class RAGService:
    def __init__(self):
        self.llm = deepseek_llm
        Settings.embed_model = embed_model
        Settings.llm = self.llm
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )

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
                logger.info(f"[rag向量存储失败]:内容为空")
                return False
            nodes = chunk_file(docs)


            VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
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
        print(f"🎯Dense检索")
        retriever = DenseRetriever(vector_store=vector_store,storage_context=self.storage_context)
        docs = retriever.run(query_result.search_queries)

        for doc in docs[:3]:
            print(doc["score"], doc["content"])
            print("***" * 50)

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
        generator = Generator(llm=self.llm)
        response = generator.run(query_result.rewrite_query, context)
        print(response)

        return response

rag_service = RAGService()

if  __name__ == "__main__":
    # data = DocumentMetadata(
    #      department_id=1,
    #      department_name="TQ",
    #      user_id=1,
    #      user_name="EdenXie",
    #      file_path="public\\uploads\\TQ\\文档上传测试.pdf",
    #      file_name= "文档上传测试.pdf",
    #      file_size=100,
    #      file_type="pdf",
    #      source="pdf"
    #  )
    # rag_service.ingestion("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\文档上传测试.pdf",data)
    rag_service.query("新格林耐特网络有限公司成立于哪一年",{})