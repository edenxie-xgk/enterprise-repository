from typing import Any

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from service.database.connect import get_session
from service.dependencies.auth import get_current_active_user
from service.models.role_department import RoleDepartmentModel
from service.models.users import UserModel
from service.router.agent.index import agent_router
from src.agent.runner import build_run_report, run_agent


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    session_id: str = Field(..., description="Chat session id")


def build_user_profile(current_user: UserModel, allowed_department_ids: list[int]) -> dict[str, Any]:
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "dept_id": current_user.dept_id,
        "department_id": current_user.dept_id,
        "role_id": current_user.role_id,
        "allowed_department_ids": allowed_department_ids,
    }


@agent_router.post("/query")
async def query_agent(
    payload: AgentQueryRequest,
    current_user: UserModel = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
):
    role_dept_result = await session.execute(
        select(RoleDepartmentModel.dept_id).where(RoleDepartmentModel.role_id == current_user.role_id)
    )
    allowed_department_ids = role_dept_result.scalars().all()

    user_profile = build_user_profile(current_user, allowed_department_ids)
    final_state = run_agent(
        payload.query,
        user_id=str(current_user.id or ""),
        session_id=payload.session_id,
        user_profile=user_profile,
        max_steps=6,
    )

    return {
        "status": final_state.status,
        "data": build_run_report(final_state),
    }
