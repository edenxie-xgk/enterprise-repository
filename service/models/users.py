import time

from sqlmodel import SQLModel, Field
from typing import Optional

from utils.utils import get_current_time


class UserModel(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password: str =  Field(index=True)
    dept_id:int = Field(index=True)
    role_id:int = Field(index=True)
    # ✅ 关键：明确用无时区的 TIMESTAMP
    create_time: str = Field(default_factory=get_current_time)
