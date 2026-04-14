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
    memory_backend: Literal["disabled", "milvus"] = Field(default=os.getenv("MEMORY_BACKEND", "disabled"))
    memory_enabled: bool = Field(default=os.getenv("MEMORY_ENABLED", "false"))
    memory_top_k: int = Field(default=int(os.getenv("MEMORY_TOP_K", 3)))
    memory_context_limit: int = Field(default=int(os.getenv("MEMORY_CONTEXT_LIMIT", 3)))
    memory_recall_min_score: float = Field(default=float(os.getenv("MEMORY_RECALL_MIN_SCORE", "0.35")))
    memory_write_enabled: bool = Field(default=os.getenv("MEMORY_WRITE_ENABLED", "false"))
    memory_write_min_importance: float = Field(default=float(os.getenv("MEMORY_WRITE_MIN_IMPORTANCE", "0.65")))
    milvus_uri: str = Field(default=os.getenv("MILVUS_URI", ""))
    milvus_token: str = Field(default=os.getenv("MILVUS_TOKEN", ""))
    milvus_db_name: str = Field(default=os.getenv("MILVUS_DB_NAME", "default"))
    milvus_collection_name: str = Field(default=os.getenv("MILVUS_COLLECTION_NAME", "long_term_memory"))
    milvus_consistency_level: str = Field(default=os.getenv("MILVUS_CONSISTENCY_LEVEL", "Bounded"))
    milvus_search_metric: str = Field(default=os.getenv("MILVUS_SEARCH_METRIC", "COSINE"))
    milvus_index_type: str = Field(default=os.getenv("MILVUS_INDEX_TYPE", "AUTOINDEX"))
    milvus_vector_dim: int = Field(default=int(os.getenv("MILVUS_VECTOR_DIM", os.getenv("EMBEDDING_DIM", "1536"))))

    mongodb_url:str = Field(default=os.getenv("MONGODB_URL"))
    mongodb_db_name:str = Field(default=os.getenv("MONGODB_DB_NAME"))
    doc_collection_name:str = Field(default=os.getenv("DOC_COLLECTION_NAME"))
    qa_collection_name:str = Field(default=os.getenv("QA_COLLECTION_NAME"))
    elasticsearch_url:str = Field(default=os.getenv("ELASTICSEARCH_URL"))

    metadata_version:int = Field(default=os.getenv("METADATA_VERSION"))
    txt_chunk_size:int = Field(default=os.getenv("TXT_CHUNK_SIZE"))
    txt_chunk_overlap:int = Field(default=os.getenv("TXT_CHUNK_OVERLAP"))
    txt_min_chunk_size:int = Field(default=os.getenv("TXT_MIN_CHUNK_SIZE"))
    docx_chunk_size:int = Field(default=os.getenv("DOCX_CHUNK_SIZE"))
    docx_chunk_overlap:int = Field(default=os.getenv("DOCX_CHUNK_OVERLAP"))
    docx_min_chunk_size:int = Field(default=os.getenv("DOCX_MIN_CHUNK_SIZE"))
    md_chunk_size:int =  Field(default=os.getenv("MD_CHUNK_SIZE"))
    md_chunk_overlap:int = Field(default=os.getenv("MD_CHUNK_OVERLAP"))
    md_min_chunk_size:int = Field(default=os.getenv("MD_MIN_CHUNK_SIZE"))
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
    agent_max_steps:int = Field(default=int(os.getenv("AGENT_MAX_STEPS", 10)))
    agent_chat_history_limit:int = Field(default=int(os.getenv("AGENT_CHAT_HISTORY_LIMIT", 8)))
    agent_output_level:Literal["concise","standard","detailed"] = Field(
        default=os.getenv("AGENT_OUTPUT_LEVEL", "standard")
    )
    monitor_default_input_cost_per_1m: float = Field(default=float(os.getenv("MONITOR_DEFAULT_INPUT_COST_PER_1M", "0")))
    monitor_default_output_cost_per_1m: float = Field(default=float(os.getenv("MONITOR_DEFAULT_OUTPUT_COST_PER_1M", "0")))
    monitor_openai_input_cost_per_1m: float = Field(default=float(os.getenv("MONITOR_OPENAI_INPUT_COST_PER_1M", "2.5")))
    monitor_openai_output_cost_per_1m: float = Field(default=float(os.getenv("MONITOR_OPENAI_OUTPUT_COST_PER_1M", "10")))
    monitor_deepseek_input_cost_per_1m: float = Field(default=float(os.getenv("MONITOR_DEEPSEEK_INPUT_COST_PER_1M", "0.27")))
    monitor_deepseek_output_cost_per_1m: float = Field(default=float(os.getenv("MONITOR_DEEPSEEK_OUTPUT_COST_PER_1M", "1.1")))
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
    zhipuai_api_key:str = Field(default=os.getenv("ZHIPUAI_API_KEY"))


settings = Settings()


if  __name__ == "__main__":
    print("***" * 50)
    print(settings.delete_file)
