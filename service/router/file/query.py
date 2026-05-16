from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from service.database.connect import get_session
from service.dependencies.auth import get_current_active_user
from service.models.department import DepartmentModel
from service.models.file import FileModel
from service.models.users import UserModel
from service.router.file.index import file_router
from service.utils.access_control import get_allowed_departments, get_allowed_department_ids
from service.utils.file_utils import build_file_download_url, get_file_state_label, is_file_ready


@file_router.get("/departments")
async def list_file_departments(
    current_user: UserModel = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
):
    departments = await get_allowed_departments(current_user=current_user, session=session)
    return {
        "code": 200,
        "message": "success",
        "data": [
            {
                "dept_id": department.dept_id,
                "dept_name": department.dept_name,
            }
            for department in departments
        ],
    }


@file_router.get("/query_file")
async def query_file(
    current_user: UserModel = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
):
    allowed_department_ids = await get_allowed_department_ids(current_user=current_user, session=session)
    if not allowed_department_ids:
        return {"code": 200, "data": [], "message": "该角色未分配部门权限"}

    file_result = await session.execute(
        select(FileModel, DepartmentModel.dept_name)
        .join(DepartmentModel, FileModel.dept_id == DepartmentModel.dept_id, isouter=True)
        .where(
            FileModel.dept_id.in_(allowed_department_ids),
            FileModel.state != "0",
        )
        .order_by(FileModel.dept_id, FileModel.create_time.desc())
    )

    rows = []
    for file_item, dept_name in file_result.all():
        download_url = build_file_download_url(int(file_item.file_id))
        rows.append(
            {
                "file_id": file_item.file_id,
                "user_id": file_item.user_id,
                "dept_id": file_item.dept_id,
                "create_time": file_item.create_time,
                "file_name": file_item.file_name,
                "file_type": file_item.file_type,
                "state": file_item.state,
                "state_label": get_file_state_label(file_item.state),
                "is_ready": is_file_ready(file_item.state),
                "dept_name": dept_name,
                "file_path": download_url,
                "download_url": download_url,
            }
        )

    return {
        "code": 200,
        "message": "success",
        "data": rows,
    }
