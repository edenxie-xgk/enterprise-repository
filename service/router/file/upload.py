import time
from typing import Tuple

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
import os

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, insert, update

from rag.ingestion.pipeline import pipeline
from service.database.connect import get_session
from service.models.department import DepartmentModel
from service.models.file import FileModel
from service.models.users import UserModel
from service.router.file.index import file_router
from core.settings import settings
from core.custom_types import DocumentMetadata
from service.utils.config import upload_dir

UPLOAD_DIR =  upload_dir


@file_router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        user_id: int = Form(...),
        dept_id: int = Form(...),
        session: AsyncSession = Depends(get_session)
):
    if not dept_id:
        return {"message": "dept_id is required"}

    if not user_id:
        return {"message": "user_id is required"}

    # 验证部门是否存在
    department_result = await session.execute(
        select(DepartmentModel).where(DepartmentModel.dept_id == dept_id)
    )
    department = department_result.first()
    if not department:
        raise HTTPException(status_code=404, detail="Department not found")

    department: DepartmentModel = department[0]

    # 验证用户是否存在
    user_result = await session.execute(
        select(UserModel).where(UserModel.id == user_id)
    )
    user = user_result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(str(UPLOAD_DIR / department.dept_name), exist_ok=True)
    file_path = str(UPLOAD_DIR / department.dept_name / file.filename)

    db_filepath = '/public/uploads/'  + department.dept_name + "/" + file.filename
    file_result = await session.execute(
        select(FileModel).where(FileModel.file_path == db_filepath, FileModel.state == '1')
    )
    res_file = file_result.first()
    if res_file:
        # ========== 改动：PostgreSQL 时间格式化调整 ==========
        old_file = res_file[0]
        time_str = str(old_file.create_time).replace(" ", "_").replace(":", "_")
        new_file_name = f"{''.join(file.filename.split('.')[:-1])}_{time_str}.{file.filename.split('.')[-1]}"
        new_file_path = str(UPLOAD_DIR / department.dept_name / new_file_name)

        os.rename(file_path, new_file_path)
        await session.execute(
            update(FileModel)
            .where(FileModel.file_path == db_filepath, FileModel.state == '1')
            .values(
                state='0',
                file_path='/public/uploads/' + department.dept_name + "/" + new_file_name,
                file_name=new_file_name
            )
        )
    file_content = await file.read()

    with open(file_path, "wb") as f:
        f.write(file_content)


    file_data = FileModel(
        user_id=user_id,
        dept_id=department.dept_id,
        file_name=file.filename,
        file_path=db_filepath,
        file_type=file.filename.split('.')[-1],
    )

    # 保存数据库 metadata
    document = DocumentMetadata(
        file_name=file.filename,
        file_path=db_filepath,
        file_size=file.size,
        file_type=file.filename.split('.')[-1],
        user_id=user_id,
        department=department.dept_name
    )

    # 进行向量数据库存(切片->存储)
    pipeline(file_content,document)



    session.add(file_data)
    await session.commit()

    # 刷新以获取数据库生成的时间戳
    await session.refresh(file_data)



    return {
        "message": "upload success",
        "document": document,
        "code": 200
    }