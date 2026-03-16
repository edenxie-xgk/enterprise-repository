import time

from llama_index.core.schema import Document
import os

from core.custom_types import DocumentMetadata
from models.embedding import embed_model
from models.llm import deepseek_llm
from utils.logger_handler import logger
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from rag.ingestion.loader import  load_file
from rag.index.vector_store import vector_store
from rag.ingestion.chunker import chunk_file


class RAGService:
    def __init__(self):
        Settings.embed_model = embed_model
        Settings.llm = deepseek_llm

    def pipeline(self, path:str, metadata: DocumentMetadata):
        """
        数据向量存储
        :param path: 文件路径
        :param metadata: 存储检索
        :return:
        """
        if not path or os.path.isdir(path):
            return False
        try:
            start_time = time.time()
            docs = load_file(path, metadata)
            if not docs:
                logger.info(f"[rag向量存储失败]:内容为空")
                return False
            nodes = chunk_file(docs)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )

            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"[rag向量存储成功]:存储文件:${metadata.file_path} 用时:${elapsed_time}s")
            return True
        except Exception as e:
            logger.error(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}")
            raise Exception(f"[rag向量存储失败]:存储文件:${metadata.file_path}---错误信息:${str(e)}") from e


rag_service = RAGService()

if  __name__ == "__main__":

    data = DocumentMetadata(
         dept_id=1,
         user_id=1,
         file_path="D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\industry_report_1.md",
         file_name= "内部讨论邮件_1.txt",
         file_size=100,
         file_type="txt"
     )
    rag_service.pipeline("D:\\python\\agent_project\\rag-agent\\service\\public\\uploads\\TQ\\industry_report_1.md",data)
