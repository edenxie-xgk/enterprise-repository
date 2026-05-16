# ruff: noqa: E402

import asyncio
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_TEST_ENV = {
    "APP_ENV": "development",
    "DELETE_FILE": "false",
    "DATABASE_NAME": "test_db",
    "DATABASE_STRING": "postgresql://user:pass@localhost:5432/test_db",
    "DATABASE_ASYNC_STRING": "postgresql+asyncpg://user:pass@localhost:5432/test_db",
    "JWT_SECRET_KEY": "test-jwt-secret",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "60",
    "CORS_ALLOW_ORIGINS": '["http://localhost:5173", "http://127.0.0.1:5173"]',
    "CORS_ALLOW_METHODS": '["GET", "POST", "PUT", "DELETE", "OPTIONS"]',
    "CORS_ALLOW_HEADERS": '["Authorization", "Content-Type", "Accept"]',
    "UPLOAD_ALLOWED_EXTENSIONS": '["txt", "pdf", "md"]',
    "UPLOAD_MAX_SIZE_MB": "5",
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from service.models import DepartmentModel, RoleDepartmentModel, RoleModel, UserModel
from service.utils.password_utils import hash_password


class FakeRAGService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ingestion(self, path: str, metadata) -> bool:
        self.calls.append(
            {
                "path": path,
                "metadata": metadata,
            }
        )
        return True


class APISmokeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_root = Path(self.temp_dir.name) / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)

        db_path = Path(self.temp_dir.name) / "api-smoke.db"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path.as_posix()}",
            future=True,
        )
        self.session_maker = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.fake_rag_service = FakeRAGService()

        await self._create_schema()
        await self._seed_data()

        self.module_patch = None
        self.upload_dir_patch = None
        self.login_token_patch = None
        self.verify_token_patch = None
        self.app = self._build_test_app()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        if self.verify_token_patch is not None:
            self.verify_token_patch.stop()
        if self.login_token_patch is not None:
            self.login_token_patch.stop()
        if self.upload_dir_patch is not None:
            self.upload_dir_patch.stop()
        if self.module_patch is not None:
            self.module_patch.stop()
        await self.engine.dispose()
        self.temp_dir.cleanup()
        self._clear_router_modules()

    async def _create_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def _seed_data(self) -> None:
        async with self.session_maker() as session:
            session.add_all(
                [
                    RoleModel(role_id=1, role_name="analyst"),
                    DepartmentModel(dept_id=1, dept_name="Research"),
                    DepartmentModel(dept_id=2, dept_name="Finance"),
                    RoleDepartmentModel(r_d_id=1, role_id=1, dept_id=1),
                    UserModel(
                        id=1,
                        username="alice",
                        password=hash_password("secret-123"),
                        dept_id=1,
                        role_id=1,
                    ),
                ]
            )
            await session.commit()

    def _build_test_app(self) -> FastAPI:
        self._clear_router_modules()

        fake_db_module = types.ModuleType("service.database.connect")
        fake_db_module.INGESTION_SEMAPHORE = asyncio.Semaphore(1)
        fake_db_module.async_session_maker = self.session_maker
        fake_db_module.async_engine = self.engine

        async def get_session():
            async with self.session_maker() as session:
                yield session

        fake_db_module.get_session = get_session

        fake_rag_module = types.ModuleType("src.rag.rag_service")
        fake_rag_module.rag_service = self.fake_rag_service

        self.module_patch = patch.dict(
            sys.modules,
            {
                "service.database.connect": fake_db_module,
                "src.rag.rag_service": fake_rag_module,
            },
        )
        self.module_patch.start()

        file_utils = importlib.import_module("service.utils.file_utils")
        self.upload_dir_patch = patch.object(file_utils, "upload_dir", self.upload_root)
        self.upload_dir_patch.start()

        user_index = importlib.import_module("service.router.users.index")
        file_index = importlib.import_module("service.router.file.index")
        login_module = importlib.import_module("service.router.users.login")
        auth_module = importlib.import_module("service.dependencies.auth")

        self.login_token_patch = patch.object(login_module, "create_access_token", return_value="test-token")
        self.login_token_patch.start()

        def fake_verify_token(token: str) -> dict:
            if token != "test-token":
                raise auth_module._credentials_exception("Could not validate credentials")
            return {
                "sub": "1",
                "username": "alice",
                "dept_id": 1,
                "role_id": 1,
            }

        self.verify_token_patch = patch.object(auth_module, "verify_token", side_effect=fake_verify_token)
        self.verify_token_patch.start()

        app = FastAPI()
        app.include_router(user_index.user_router)
        app.include_router(file_index.file_router)
        app.include_router(file_index.legacy_public_file_router)
        return app

    @staticmethod
    def _clear_router_modules() -> None:
        module_names = [
            "service.database.connect",
            "service.dependencies.auth",
            "service.router.users.index",
            "service.router.users.login",
            "service.router.users.profile",
            "service.router.file.index",
            "service.router.file.upload",
            "service.router.file.query",
            "service.router.file.download",
            "src.rag.rag_service",
        ]
        for module_name in module_names:
            sys.modules.pop(module_name, None)

    async def _login(self) -> str:
        response = await self.client.post(
            "/user/login",
            json={
                "username": "alice",
                "password": "secret-123",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        return payload["data"]["access_token"]

    async def test_login_returns_access_token_and_user_info(self):
        response = await self.client.post(
            "/user/login",
            json={
                "username": "alice",
                "password": "secret-123",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["code"], 200)
        self.assertEqual(payload["data"]["token_type"], "bearer")
        self.assertEqual(payload["data"]["user"]["id"], 1)
        self.assertEqual(payload["data"]["user"]["dept_id"], 1)
        self.assertTrue(payload["data"]["access_token"])

    async def test_file_query_requires_bearer_token(self):
        response = await self.client.get("/file/query_file")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Missing bearer token")

    async def test_upload_query_and_download_flow_uses_authenticated_user(self):
        token = await self._login()
        headers = {"Authorization": f"Bearer {token}"}

        upload_response = await self.client.post(
            "/file/upload",
            headers=headers,
            data={
                "dept_id": "1",
                "user_id": "999",
            },
            files={
                "file": (
                    "../Quarterly:Report?.pdf",
                    b"quarterly report content",
                    "application/pdf",
                )
            },
        )

        self.assertEqual(upload_response.status_code, 200)
        upload_payload = upload_response.json()
        self.assertEqual(upload_payload["code"], 200)
        self.assertEqual(upload_payload["data"]["file_name"], "Quarterly_Report_.pdf")
        self.assertEqual(upload_payload["data"]["dept_id"], 1)
        self.assertEqual(upload_payload["data"]["dept_name"], "Research")
        self.assertEqual(upload_payload["data"]["state"], "3")
        self.assertEqual(upload_payload["data"]["state_label"], "待处理")
        self.assertFalse(upload_payload["data"]["is_ready"])
        self.assertEqual(upload_payload["data"]["file_path"], "/file/files/1/download")
        self.assertEqual(upload_payload["data"]["download_url"], "/file/files/1/download")
        self.assertEqual(len(self.fake_rag_service.calls), 1)
        self.assertEqual(self.fake_rag_service.calls[0]["metadata"].user_id, 1)
        self.assertEqual(self.fake_rag_service.calls[0]["metadata"].department_id, 1)

        query_response = await self.client.get("/file/query_file", headers=headers)
        self.assertEqual(query_response.status_code, 200)
        query_payload = query_response.json()
        self.assertEqual(query_payload["code"], 200)
        self.assertEqual(len(query_payload["data"]), 1)
        file_row = query_payload["data"][0]
        self.assertEqual(file_row["user_id"], 1)
        self.assertEqual(file_row["dept_id"], 1)
        self.assertEqual(file_row["file_name"], "Quarterly_Report_.pdf")
        self.assertEqual(file_row["state"], "1")
        self.assertEqual(file_row["state_label"], "已就绪")
        self.assertTrue(file_row["is_ready"])
        self.assertEqual(file_row["download_url"], "/file/files/1/download")

        download_response = await self.client.get(file_row["download_url"], headers=headers)
        self.assertEqual(download_response.status_code, 200)
        self.assertEqual(download_response.content, b"quarterly report content")

    async def test_upload_rejects_department_outside_role_scope(self):
        token = await self._login()
        headers = {"Authorization": f"Bearer {token}"}

        response = await self.client.post(
            "/file/upload",
            headers=headers,
            data={"dept_id": "2"},
            files={
                "file": (
                    "notes.pdf",
                    b"unauthorized department upload",
                    "application/pdf",
                )
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "You do not have access to the requested department")


if __name__ == "__main__":
    unittest.main()
