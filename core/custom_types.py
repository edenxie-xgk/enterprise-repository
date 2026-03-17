import time

from pydantic import BaseModel

from core.settings import settings


class DocumentMetadata(BaseModel):
    """文档元数据结构。"""
    file_name:str
    file_path:str
    file_type:str
    file_size:int
    source:str = None

    section_title:str = None
    page:int = None

    user_id: int
    user_name: str
    department_id:int
    department_name:str

    version:int = settings.metadata_version


