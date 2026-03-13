from fastapi import APIRouter


user_router = APIRouter(prefix="/user", tags=["用户模块"])

from . import login