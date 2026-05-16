from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from fastapi import HTTPException, status

from core.settings import settings
from service.utils.config import upload_dir

_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]+')
FILE_STATE_LABELS = {
    "0": "已删除",
    "1": "已就绪",
    "2": "处理中",
    "3": "待处理",
    "4": "失败",
}
READY_FILE_STATES = {"1"}


def get_file_state_label(state: str | None) -> str:
    return FILE_STATE_LABELS.get(str(state or ""), str(state or "未知"))


def is_file_ready(state: str | None) -> bool:
    return str(state or "") in READY_FILE_STATES


def sanitize_filename(filename: str) -> str:
    raw_name = Path(filename or "").name.strip()
    cleaned = _INVALID_FILENAME_CHARS.sub("_", raw_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file name",
        )
    return cleaned


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def ensure_upload_is_allowed(*, file_name: str, file_size: int) -> None:
    extension = get_file_extension(file_name)
    allowed_extensions = {item.lower() for item in settings.upload_allowed_extensions}
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {extension or 'unknown'}",
        )

    if file_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    max_size_bytes = settings.upload_max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File is too large. Max size is {settings.upload_max_size_mb} MB",
        )


def build_file_download_url(file_id: int) -> str:
    return f"/file/files/{file_id}/download"


def build_legacy_public_file_path(department_name: str, file_name: str) -> str:
    return f"/public/uploads/{department_name}/{file_name}"


def resolve_storage_path(*, department_name: str, file_name: str) -> Path:
    return upload_dir / department_name / file_name


def build_archived_file_name(*, original_file_name: str, create_time: str | None) -> str:
    path = Path(original_file_name)
    suffix = path.suffix
    stem = path.stem or "file"
    timestamp = (create_time or "").strip().replace(" ", "_").replace(":", "_")
    timestamp = timestamp or "archived"
    return f"{stem}_{timestamp}{suffix}"


def guess_media_type(file_name: str) -> str:
    media_type, _ = mimetypes.guess_type(file_name)
    return media_type or "application/octet-stream"
