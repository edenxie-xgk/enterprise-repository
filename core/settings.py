import logging
import os
import pathlib
from typing import Literal

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

dotenv.load_dotenv()
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

class Settings(BaseSettings):
    #文件相关的
    root_dir:pathlib.Path = Field(default=pathlib.Path(__file__).parent.parent)
    delete_file:bool = Field(default=os.getenv("DELETE_FILE"))


    #PostgerSQL
    database_name:str = Field(default= os.getenv("DATABASE_NAME"))
    database_string:str =  Field(default=str(os.getenv("DATABASE_STRING")))
    database_async_string:str =  Field(default=str(os.getenv("DATABASE_ASYNC_STRING")))
    vector_table_name:str = Field(default=os.getenv("VECTOR_TABLE_NAME"))
    embedding_model:str = Field(default=os.getenv("EMBEDDING_MODEL"))
    embedding_dim:int = Field(default=os.getenv("EMBEDDING_DIM"))

    mongodb_url:str = Field(default=os.getenv("MONGODB_URL"))
    mongodb_db_name:str = Field(default=os.getenv("MONGODB_DB_NAME"))
    doc_collection_name:str = Field(default=os.getenv("DOC_COLLECTION_NAME"))
    qa_collection_name:str = Field(default=os.getenv("QA_COLLECTION_NAME"))
    elasticsearch_url:str = Field(default=os.getenv("ELASTICSEARCH_URL"))

    metadata_version:int = Field(default=os.getenv("METADATA_VERSION"))
    txt_chunk_size:int = Field(default=os.getenv("TXT_CHUNK_SIZE"))
    txt_chunk_overlap:int = Field(default=os.getenv("TXT_CHUNK_OVERLAP"))
    docx_chunk_size:int = Field(default=os.getenv("DOCX_CHUNK_SIZE"))
    docx_chunk_overlap:int = Field(default=os.getenv("DOCX_CHUNK_OVERLAP"))
    md_chunk_size:int =  Field(default=os.getenv("MD_CHUNK_SIZE"))
    md_chunk_overlap:int = Field(default=os.getenv("MD_CHUNK_OVERLAP"))
    pdf_chunk_size:int = Field(default=os.getenv("PDF_CHUNK_SIZE"))
    pdf_chunk_overlap:int = Field(default=os.getenv("PDF_CHUNK_OVERLAP"))
    orc_lang:str = Field(default=os.getenv("OCR_LANG"))
    orc_min_score:float = Field(default=os.getenv("OCR_MIN_SCORE"))
    excel_chunk_size:int = Field(default=os.getenv("EXCEL_CHUNK_SIZE"))
    excel_min_chunk_size:int = Field(default=os.getenv("EXCEL_MIN_CHUNK_SIZE"))
    excel_chunk_overlap:int = Field(default=os.getenv("EXCEL_CHUNK_OVERLAP"))
    excel_header_mode:str = Field(default=os.getenv("EXCEL_HEADER_MODE"))
    pptx_chunk_size:int = Field(default=os.getenv("PPTX_CHUNK_SIZE"))
    pptx_chunk_overlap:int = Field(default=os.getenv("PPTX_CHUNK_OVERLAP"))
    pptx_filter_slider:list[str] = Field(default=["thanks", "致谢", "thank you"])
    json_chunk_size:int = Field(default=os.getenv("JSON_CHUNK_SIZE"))
    json_chunk_overlap:int = Field(default=os.getenv("JSON_CHUNK_OVERLAP"))
    json_min_chunk_size:int = Field(default=os.getenv("JSON_MIN_CHUNK_SIZE"))
    image_chunk_size:int = Field(default=os.getenv("IMAGE_CHUNK_SIZE"))
    image_chunk_overlap:int = Field(default=os.getenv("IMAGE_CHUNK_OVERLAP"))

    retriever_top_k:int = Field(default=os.getenv("RETRIEVER_TOP_K"))
    reranker_top_k:int = Field(default=os.getenv("RERANKER_TOP_K"))
    reranker_type:Literal["llm",'cross-encoder'] = Field(default=os.getenv("RERANKER_TYPE"))
    bm25_retrieval_mode:Literal["lite",'es'] = Field(default=os.getenv("BM25_RETRIEVAL_MODE"))
    reranker_max_len:int = Field(default=os.getenv("RERANKER_MAX_LEN"))
    retrieval_min_score:float = Field(default=os.getenv("RETRIEVAL_MIN_SCORE"))
    reranker_min_score:float = Field(default=os.getenv("RERANKER_MIN_SCORE"))
    context_max_len:int = Field(default=os.getenv("CONTEXT_MAX_LEN"))
    max_expand:int = Field(default=os.getenv("MAX_EXPAND"))
    update_doc_time:int = Field(default=os.getenv("UPDATE_DOC_TIME"))
    is_need_doc:bool = Field(default=False)
    await_upload_file_num:int = Field(default=0)

    log_dir:pathlib.Path = Field(default=pathlib.Path(__file__).parent.parent / "logs")
    log_format:logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    max_retries:int = Field(default=os.getenv("MAX_RETRIES"))
    max_timeout:int = Field(default=os.getenv("MAX_TIMEOUT"))
    hf_token:str = Field(default=os.getenv("HF_TOKEN"))
    reranker_model:str = Field(default=os.getenv("RERANKER_MODEL"))
    openai_api_key:str=Field(default=os.getenv("OPENAI_API_KEY"))
    openai_model:str=Field(default=os.getenv("OPENAI_MODEL"))
    openai_base_url:str=Field(default=os.getenv("OPENAI_BASE_URL"))
    deepseek_base_url:str = Field(default=os.getenv("DEEPSEEK_URL"))
    deepseek_model:str = Field(default=os.getenv("DEEPSEEK_MODEL"))
    deepseek_api_key:str = Field(default=os.getenv("DEEPSEEK_API_KEY"))


settings = Settings()


if  __name__ == "__main__":
    print("***" * 50)
    print(settings.delete_file)
