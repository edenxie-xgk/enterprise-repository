from sqlmodel import SQLModel, Field


class  RoleDepartmentModel(SQLModel, table=True):
    __tablename__ = "role_department"

    r_d_id:int = Field(primary_key=True)
    role_id: int = Field(index=True)
    dept_id: int = Field(index=True)

