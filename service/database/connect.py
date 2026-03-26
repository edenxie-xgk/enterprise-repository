import asyncio

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.settings import settings

# PostgreSQL 异步连接字符串格式：postgresql+asyncpg://user:password@host:port/dbname
DATABASE_URL = settings.database_async_string

# 创建异步引擎
# echo=True 会打印 SQL 语句，pool_pre_ping=True 保持连接活跃
async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    pool_size=20,  # 从 5 增大到 20
    max_overflow=30,  # 从 10 增大到 30
    pool_timeout=60,  # 等待连接的超时时间
    pool_recycle=3600,  # 连接回收时间
    pool_pre_ping=True,  # 连接前 ping 一下, 避免用死连
)
# 全局信号量, 控制同时处理的文件数
INGESTION_SEMAPHORE = asyncio.Semaphore(5)  # 最多5个文件同时入库
# 创建异步会话工厂
async_session_maker = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# 依赖注入：获取 DB Session
async def get_session():
    async with async_session_maker() as session:
        yield session