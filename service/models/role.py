from sqlmodel import SQLModel, Field


class RoleModel(SQLModel, table=True):
    __tablename__ = "role"

    role_id:int = Field(primary_key=True)
    role_name:str = Field(index=True)