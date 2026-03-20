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
    hf_token:str = Field(default=os.getenv("HF_TOKEN"))
    embedding_model:str = Field(default=os.getenv("EMBEDDING_MODEL"))
    embedding_dim:int = Field(default=os.getenv("EMBEDDING_DIM"))

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
    reranker_max_len:int = Field(default=os.getenv("RERANKER_MAX_LEN"))
    reranker_min_score:float = Field(default=os.getenv("RERANKER_MIN_SCORE"))
    context_max_len:int = Field(default=os.getenv("CONTEXT_MAX_LEN"))

    log_dir:pathlib.Path = Field(default=pathlib.Path(__file__).parent.parent / "logs")
    log_format:logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
