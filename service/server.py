from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from core.settings import settings
from service.database_initializer import initialize_project_database
from service.database.connect import async_engine
from service.router.agent.index import agent_router
from service.router.file.index import file_router, legacy_public_file_router
from service.router.role.index import role_router
from service.router.users.index import user_router


def create_server() -> FastAPI:
    settings.validate_runtime_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Application startup...")
        schema_mode = None
        if settings.database_auto_migrate:
            schema_mode = "migrate"
            print("DATABASE_AUTO_MIGRATE is enabled. Applying Alembic migrations on startup.")
        elif settings.database_auto_create:
            schema_mode = "create_all"
            print("DATABASE_AUTO_CREATE is enabled. Using metadata.create_all() on startup.")
        else:
            print(
                "Schema auto-init flags are disabled. Startup will verify schema and, when configured, "
                "auto-bootstrap missing tables on first start."
            )

        await initialize_project_database(
            schema_mode=schema_mode,
            ensure_seed=True,
            fail_if_schema_missing=False,
        )

        yield

        print("Application shutdown...")
        await async_engine.dispose()

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    public_dir = settings.resolved_public_dir
    if settings.serve_public_files and public_dir.exists():
        app.mount(settings.normalized_public_url_path, StaticFiles(directory=str(public_dir)), name="public")

    app.include_router(router=file_router)
    app.include_router(router=legacy_public_file_router)
    app.include_router(router=agent_router)
    app.include_router(router=user_router)
    app.include_router(router=role_router)
    return app


if __name__ == "__main__":
    uvicorn.run(create_server(), host=settings.server_host, port=settings.server_port)
