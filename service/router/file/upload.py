import asyncio

from fastapi import BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, update

from core.custom_types import DocumentMetadata
from core.settings import settings
from service.database.connect import INGESTION_SEMAPHORE, async_session_maker, get_session
from service.dependencies.auth import get_current_active_user
from service.models.file import FileModel
from service.models.users import UserModel
from service.router.file.index import file_router
from service.utils.access_control import resolve_authorized_department
from service.utils.file_utils import (
    build_archived_file_name,
    build_file_download_url,
    ensure_upload_is_allowed,
    get_file_state_label,
    get_file_extension,
    is_file_ready,
    resolve_storage_path,
    sanitize_filename,
)
from src.rag.rag_service import rag_service
from utils.logger_handler import logger


@file_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    dept_id: int | None = Form(default=None),
    user_id: int | None = Form(default=None),
    background_tasks: BackgroundTasks = None,
    current_user: UserModel = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
):
    del user_id  # Compatibility field for old clients; auth is derived from the bearer token.

    if background_tasks is None:
        raise HTTPException(status_code=500, detail="Background task manager is unavailable")

    department, _ = await resolve_authorized_department(
        current_user=current_user,
        session=session,
        requested_department_id=dept_id,
    )

    safe_file_name = sanitize_filename(file.filename or "")
    file_content = await file.read()
    file_size = len(file_content)
    ensure_upload_is_allowed(file_name=safe_file_name, file_size=file_size)

    department_storage_path = resolve_storage_path(
        department_name=department.dept_name,
        file_name="placeholder",
    ).parent
    department_storage_path.mkdir(parents=True, exist_ok=True)
    storage_path = resolve_storage_path(
        department_name=department.dept_name,
        file_name=safe_file_name,
    )

    existing_result = await session.execute(
        select(FileModel).where(
            FileModel.dept_id == department.dept_id,
            FileModel.file_name == safe_file_name,
            FileModel.state == "1",
        )
    )
    existing_file = existing_result.scalar_one_or_none()
    if existing_file is not None:
        if not settings.delete_file:
            archived_file_name = build_archived_file_name(
                original_file_name=existing_file.file_name,
                create_time=existing_file.create_time,
            )
            archived_storage_path = resolve_storage_path(
                department_name=department.dept_name,
                file_name=archived_file_name,
            )
            if storage_path.exists() and not archived_storage_path.exists():
                storage_path.rename(archived_storage_path)
            existing_file.file_name = archived_file_name
        elif storage_path.exists():
            storage_path.unlink()

        existing_file.state = "0"
        session.add(existing_file)

    file_data = FileModel(
        user_id=int(current_user.id),
        dept_id=department.dept_id,
        file_name=safe_file_name,
        file_path="pending",
        file_type=get_file_extension(safe_file_name),
    )
    session.add(file_data)
    await session.flush()

    file_id = int(file_data.file_id)
    file_data.file_path = build_file_download_url(file_id)

    document = DocumentMetadata(
        file_name=safe_file_name,
        file_path=file_data.file_path,
        file_size=file_size,
        file_type=file_data.file_type,
        source=file_data.file_type,
        user_id=int(current_user.id),
        user_name=current_user.username,
        department_id=department.dept_id,
        department_name=department.dept_name,
    )

    try:
        with open(storage_path, "wb") as storage_file:
            storage_file.write(file_content)

        await session.commit()
        await session.refresh(file_data)

        async def controlled_ingestion() -> None:
            async with INGESTION_SEMAPHORE:
                async with async_session_maker() as worker_session:
                    try:
                        settings.await_upload_file_num += 1
                        await worker_session.execute(
                            update(FileModel)
                            .where(FileModel.file_id == file_id)
                            .values(state="2")
                        )
                        await worker_session.commit()

                        is_success = await asyncio.to_thread(
                            rag_service.ingestion,
                            str(storage_path),
                            document,
                        )

                        await worker_session.execute(
                            update(FileModel)
                            .where(FileModel.file_id == file_id)
                            .values(state="1" if is_success else "4")
                        )
                        await worker_session.commit()

                        if is_success:
                            settings.is_need_doc = True
                    except Exception as exc:
                        logger.error(f"Ingestion failed: {exc}")
                        try:
                            await worker_session.execute(
                                update(FileModel)
                                .where(FileModel.file_id == file_id)
                                .values(state="4")
                            )
                            await worker_session.commit()
                        except Exception:
                            pass
                    finally:
                        settings.await_upload_file_num = max(0, settings.await_upload_file_num - 1)

        background_tasks.add_task(controlled_ingestion)
        return {
            "code": 200,
            "message": "upload success",
            "data": {
                "file_id": file_id,
                "file_name": file_data.file_name,
                "dept_id": file_data.dept_id,
                "dept_name": department.dept_name,
                "state": file_data.state,
                "state_label": get_file_state_label(file_data.state),
                "is_ready": is_file_ready(file_data.state),
                "file_path": file_data.file_path,
                "download_url": file_data.file_path,
            },
        }
    except Exception as exc:
        if storage_path.exists():
            storage_path.unlink()
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
