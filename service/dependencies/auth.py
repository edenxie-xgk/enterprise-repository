from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from service.database.connect import get_session
from service.models.users import UserModel
from service.utils.jwt_utils import verify_token

# 使用 HTTPBearer 提取 Token
security = HTTPBearer()


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        session: AsyncSession = Depends(get_session)
) -> UserModel:
    """
    验证 Token 并返回当前登录用户
    用法：在路由函数中添加参数 current_user: UserModel = Depends(get_current_user)
    """
    token = credentials.credentials
    payload = verify_token(token)

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: user_id not found"
        )

    # 查询用户信息
    result = await session.execute(
        select(UserModel).where(UserModel.id == int(user_id))
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user


async def get_current_active_user(
        current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    """
    可以扩展：检查用户是否被禁用等额外验证
    """
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user