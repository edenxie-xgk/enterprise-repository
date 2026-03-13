from typing import Optional

from sqlmodel import SQLModel, Field


class DepartmentModel(SQLModel, table=True):
    __tablename__ = "department"

    dept_id: Optional[int] = Field(default=None, primary_key=True)
    dept_name: str = Field(index=True, unique=True)