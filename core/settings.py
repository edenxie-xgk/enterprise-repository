import logging
import os
import pathlib
import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):
    #文件相关的
    root_dir:pathlib.Path = Field(default=pathlib.Path(__file__).parent.parent)

    #PostgerSQL
    database_name:str = Field(default= os.getenv("DATABASE_NAME"))
    database_string:str =  Field(default=str(os.getenv("DATABASE_STRING")+'/'+os.getenv("DATABASE_NAME")))
    database_async_string:str =  Field(default=str(os.getenv("DATABASE_ASYNC_STRING")+'/'+os.getenv("DATABASE_NAME")))
    vector_table_name:str = Field(default=os.getenv("VECTOR_TABLE_NAME"))
    hf_token:str = Field(default=os.getenv("HF_TOKEN"))
    embedding_name:str = Field(default=os.getenv("EMBEDDING_NAME"))
    embedding_dim:str = Field(default=os.getenv("EMBEDDING_DIM"))

    log_dir:pathlib.Path = Field(default=pathlib.Path(__file__).parent.parent / "logs")
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


settings = Settings()


if  __name__ == "__main__":
    print("***" * 50)
    print(settings.root_dir)
