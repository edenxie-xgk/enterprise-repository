import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlmodel import SQLModel
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from service.database.connect import async_engine
from service.router.agent.index import agent_router
from service.router.file.index import file_router
import uvicorn

from service.router.role.index import role_router
from service.router.users.index import user_router




def create_server():
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("🚀 Application startup...")
        async with async_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        yield

        print("🛑 Application shutdown...")
        await async_engine.dispose()

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.join(current_dir, "public")

    # 挂载静态文件，访问路径为 /static/文件名
    app.mount("/public", StaticFiles(directory=public_dir), name="static")

    app.include_router(router=file_router)
    app.include_router(router=agent_router)
    app.include_router(router=user_router)
    app.include_router(router=role_router)
    uvicorn.run(app, host="127.0.0.1", port=1016)


if __name__ == "__main__":
    create_server()
