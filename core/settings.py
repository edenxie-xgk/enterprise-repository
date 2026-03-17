import logging
import os
import pathlib
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
    embedding_name:str = Field(default=os.getenv("EMBEDDING_NAME"))
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

    log_dir:pathlib.Path = Field(default=pathlib.Path(__file__).parent.parent / "logs")
    log_format:logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    deepseek_url:str = Field(default=os.getenv("DEEPSEEK_URL"))
    deepseek_model:str = Field(default=os.getenv("DEEPSEEK_MODEL"))
    deepseek_key:str = Field(default=os.getenv("DEEPSEEK_KEY"))

settings = Settings()


if  __name__ == "__main__":
    print("***" * 50)
    print(settings.delete_file)
