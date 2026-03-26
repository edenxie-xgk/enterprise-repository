import time
from typing import Optional

# 新增导入
from sqlmodel import SQLModel, Field, Relationship

from utils.utils import get_current_time


class FileModel(SQLModel, table=True):
    __tablename__ = 'file'

    file_id: int = Field(primary_key=True)
    user_id: int = Field(index=True)
    dept_id: int = Field(index=True)

    create_time: str = Field(default_factory=get_current_time)
    file_name: str = Field(index=True)
    file_path: str = Field(index=True)
    file_type: str = Field(index=True)
    state: str = Field(default='3', index=True)