from fastapi.params import Form, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from service.database.connect import get_session
from service.dependencies.auth import get_current_active_user
from service.models.department import DepartmentModel
from service.models.file import FileModel
from service.models.role_department import RoleDepartmentModel
from service.models.users import UserModel
from service.router.file.index import file_router

@file_router.get("/query_file")
async def query_file(
        current_user: UserModel = Depends(get_current_active_user),
        session:AsyncSession = Depends(get_session)
):

    role_dept_result = await session.execute(
        select(RoleDepartmentModel.dept_id).where(
            RoleDepartmentModel.role_id == current_user.role_id
        )
    )
    dept_ids = role_dept_result.scalars().all()  # ✅ 关键：用 .all() 收集结果
    # 检查是否有部门权限
    if not dept_ids:
        return {"code": 200, "data": [], "message": "该角色未分配部门权限"}
    file_result = await session.execute(
        select(*[getattr(FileModel, c.name) for c in FileModel.__table__.columns],DepartmentModel.dept_name).join(DepartmentModel,FileModel.dept_id == DepartmentModel.dept_id,isouter=True).where(
            FileModel.dept_id.in_(dept_ids),
            FileModel.state == '1'
        )
    )
    return {
        "code":200,
        "data":file_result.mappings().all()
    }





