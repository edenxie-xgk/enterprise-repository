from fastapi import APIRouter

file_router = APIRouter(prefix="/file", tags=["文件模块"])

from . import upload,query