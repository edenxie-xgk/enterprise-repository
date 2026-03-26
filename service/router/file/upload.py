import asyncio
from functools import partial

from fastapi import UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
import os

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, update

from core.settings import settings
from src.rag.rag_service import rag_service
from service.database.connect import get_session, async_session_maker, INGESTION_SEMAPHORE
from service.models.department import DepartmentModel
from service.models.file import FileModel
from service.models.users import UserModel
from service.router.file.index import file_router
from core.custom_types import DocumentMetadata
from service.utils.config import upload_dir
from utils.logger_handler import  logger
UPLOAD_DIR =  upload_dir


@file_router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        user_id: int = Form(...),
        dept_id: int = Form(...),
        background_tasks: BackgroundTasks = None,
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

    user: UserModel = user[0]

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(str(UPLOAD_DIR / department.dept_name), exist_ok=True)
    file_path = str(UPLOAD_DIR / department.dept_name / file.filename)

    db_filepath = '/public/uploads/'  + department.dept_name + "/" + file.filename
    file_result = await session.execute(
        select(FileModel).where(FileModel.file_path == db_filepath, FileModel.state == '1')
    )
    res_file = file_result.first()
    if res_file:
        # 是否覆盖旧文件
        if not settings.delete_file:
            old_file = res_file[0]
            time_str = str(old_file.create_time).replace(" ", "_").replace(":", "_")
            new_file_name = f"{''.join(file.filename.split('.')[:-1])}_{time_str}.{file.filename.split('.')[-1]}"
            new_file_path = str(UPLOAD_DIR / department.dept_name / new_file_name)
            if not os.path.exists(new_file_path) and os.path.exists(file_path):
                os.rename(file_path, new_file_path)
            await session.execute(
                update(FileModel)
                .where(FileModel.file_path == db_filepath, FileModel.state != '0')
                .values(
                    state='0',
                    file_path='/public/uploads/' + department.dept_name + "/" + new_file_name,
                    file_name=new_file_name
                )
            )
        else:
            await session.execute(
                update(FileModel)
                .where(FileModel.file_path == db_filepath, FileModel.state == '1')
                .values(
                    state='0',
                )
            )
    file_content = await file.read()

    # 保存数据库 metadata
    document = DocumentMetadata(
        file_name=file.filename,
        file_path=db_filepath,
        file_size=file.size,
        file_type=file.filename.split('.')[-1],
        source=file.filename.split('.')[-1],
        user_id=user_id,
        user_name=user.username,
        department_id=department.dept_id,
        department_name=department.dept_name
    )



    # 数据库存储
    file_data = FileModel(
        user_id=user_id,
        dept_id=department.dept_id,
        file_name=file.filename,
        file_path=db_filepath,
        file_type=file.filename.split('.')[-1],
    )
    session.add(file_data)

    # 文件写入
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:

        await session.commit()
        # 刷新以获取数据库生成的时间戳
        await session.refresh(file_data)

        file_id = file_data.file_id

        # 后台任务 - 关键修复在这里
        async def controlled_ingestion():
            """用信号量限制并发, 避免连接池耗尽"""
            async with INGESTION_SEMAPHORE:  # ← 关键: 限制并发
                async with async_session_maker() as s:  # 上下文管理确保释放
                    try:
                        settings.await_upload_file_num +=1
                        # 先更新为处理中状态
                        await s.execute(
                            update(FileModel)
                            .where(FileModel.file_id == file_id)
                            .values(state='2')  # processing
                        )
                        await s.commit()
                        # 执行 ingestion (在线程池, 但信号量限制了并发数)
                        is_success = await asyncio.to_thread(
                            rag_service.ingestion,
                            file_path,
                            document
                        )

                        # 更新最终状态
                        final_state = '1' if is_success else '4'
                        await s.execute(
                            update(FileModel)
                            .where(FileModel.file_id == file_id)
                            .values(state=final_state)
                        )
                        await s.commit()

                        if is_success:
                            settings.is_need_doc = True
                        settings.await_upload_file_num -=1
                    except Exception as e:
                        logger.error(f"Ingestion failed: {e}")
                        settings.await_upload_file_num -= 1
                        try:
                            await s.execute(
                                update(FileModel)
                                .where(FileModel.file_id == file_id)
                                .values(state='4')
                            )
                            await s.commit()
                        except:
                            pass
        # 添加到后台任务
        background_tasks.add_task(controlled_ingestion)

        return {"message": "upload success", "file_id": file_id, "code": 200}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

