from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from alembic import command
from alembic.config import Config
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, inspect
from sqlmodel import SQLModel, select

import service.models  # noqa: F401
from core.settings import settings
from service.database.connect import async_engine, async_session_maker
from service.models.department import DepartmentModel
from service.models.role import RoleModel
from service.models.role_department import RoleDepartmentModel
from service.models.user_profile import UserProfileModel
from service.models.users import UserModel
from service.utils.password_utils import hash_password, verify_password
from service.utils.user_types import ADMIN_USER_TYPE, NORMAL_USER_TYPE, normalize_user_type
from utils.utils import get_current_time


logger = logging.getLogger(__name__)

SchemaMode = Literal["migrate", "create_all", "auto"]


class SeedDepartment(BaseModel):
    dept_id: int
    dept_name: str = Field(min_length=1, max_length=128)


class SeedRole(BaseModel):
    role_id: int
    role_name: str = Field(min_length=1, max_length=128)
    dept_ids: list[int] = Field(default_factory=list)


class SeedUser(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    dept_id: int
    role_id: int
    user_type: str = Field(default=NORMAL_USER_TYPE)
    update_password_on_bootstrap: bool = True
    answer_style: str = "standard"
    preferred_language: str = "zh-CN"
    preferred_topics: list[str] = Field(default_factory=list)
    prefers_citations: bool = True
    allow_web_search: bool = False
    profile_notes: str = ""


class SeedDocument(BaseModel):
    departments: list[SeedDepartment] = Field(default_factory=list)
    roles: list[SeedRole] = Field(default_factory=list)
    users: list[SeedUser] = Field(default_factory=list)


def _bootstrap_enabled() -> bool:
    return bool(settings.bootstrap_seed_enabled or settings.bootstrap_admin_enabled)


def _load_seed_items(raw_value: str | None, model_cls: type[BaseModel], env_name: str) -> list[Any]:
    if raw_value is None:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{env_name} must be valid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise RuntimeError(f"{env_name} must be a JSON array.")

    items: list[Any] = []
    for index, item in enumerate(parsed):
        try:
            items.append(model_cls.model_validate(item))
        except ValidationError as exc:
            raise RuntimeError(f"{env_name}[{index}] is invalid: {exc}") from exc
    return items


def _resolve_seed_file_path(raw_value: str | None) -> Path | None:
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized:
        return None

    path = Path(normalized).expanduser()
    if not path.is_absolute():
        path = settings.root_dir / path
    return path


def _load_seed_document(seed_file: str | None) -> SeedDocument | None:
    path = _resolve_seed_file_path(seed_file)
    if path is None:
        return None
    if not path.is_file():
        raise RuntimeError(f"BOOTSTRAP_SEED_FILE not found: {path}")

    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read BOOTSTRAP_SEED_FILE: {path}") from exc

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"BOOTSTRAP_SEED_FILE must be valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("BOOTSTRAP_SEED_FILE must be a JSON object with departments, roles, and users keys.")

    try:
        return SeedDocument.model_validate(parsed)
    except ValidationError as exc:
        raise RuntimeError(f"BOOTSTRAP_SEED_FILE is invalid: {exc}") from exc


def _build_default_seed_departments() -> list[SeedDepartment]:
    return [
        SeedDepartment(
            dept_id=settings.bootstrap_admin_dept_id,
            dept_name=settings.bootstrap_admin_dept_name,
        )
    ]


def _build_default_seed_roles() -> list[SeedRole]:
    return [
        SeedRole(
            role_id=settings.bootstrap_admin_role_id,
            role_name=settings.bootstrap_admin_role_name,
            dept_ids=[settings.bootstrap_admin_dept_id],
        )
    ]


def _build_default_seed_users() -> list[SeedUser]:
    return [
        SeedUser(
            username=settings.bootstrap_admin_username,
            password=settings.bootstrap_admin_password,
            dept_id=settings.bootstrap_admin_dept_id,
            role_id=settings.bootstrap_admin_role_id,
            user_type=ADMIN_USER_TYPE,
            update_password_on_bootstrap=True,
        )
    ]


def load_bootstrap_plan() -> tuple[list[SeedDepartment], list[SeedRole], list[SeedUser]]:
    seed_document = _load_seed_document(settings.bootstrap_seed_file)
    departments = _load_seed_items(
        settings.bootstrap_seed_departments_json,
        SeedDepartment,
        "BOOTSTRAP_SEED_DEPARTMENTS_JSON",
    )
    roles = _load_seed_items(
        settings.bootstrap_seed_roles_json,
        SeedRole,
        "BOOTSTRAP_SEED_ROLES_JSON",
    )
    users = _load_seed_items(
        settings.bootstrap_seed_users_json,
        SeedUser,
        "BOOTSTRAP_SEED_USERS_JSON",
    )

    if settings.bootstrap_seed_departments_json is None:
        if seed_document is not None and "departments" in seed_document.model_fields_set:
            departments = list(seed_document.departments)
        else:
            departments = _build_default_seed_departments()
    if settings.bootstrap_seed_roles_json is None:
        if seed_document is not None and "roles" in seed_document.model_fields_set:
            roles = list(seed_document.roles)
        else:
            roles = _build_default_seed_roles()
    if settings.bootstrap_seed_users_json is None:
        if seed_document is not None and "users" in seed_document.model_fields_set:
            users = list(seed_document.users)
        else:
            users = _build_default_seed_users()

    return departments, roles, users


async def _ensure_schema_if_missing_after_skipped_init() -> tuple[str, bool]:
    schema_ready = await has_core_schema()
    if schema_ready:
        return "skipped", False

    if not settings.database_auto_init_on_startup:
        return "skipped", False

    logger.warning(
        "Core schema is missing and DATABASE_AUTO_INIT_ON_STARTUP is enabled. Applying automatic first-start initialization."
    )
    schema_action = await apply_schema("auto")
    return schema_action, True


async def _resolve_schema_readiness(
    *,
    initial_schema_action: str,
) -> tuple[bool, str, bool]:
    schema_ready = await has_core_schema()
    if schema_ready or initial_schema_action != "skipped":
        return schema_ready, initial_schema_action, False

    schema_action, auto_initialized = await _ensure_schema_if_missing_after_skipped_init()
    schema_ready = await has_core_schema()
    return schema_ready, schema_action, auto_initialized


async def initialize_project_database(
    *,
    schema_mode: SchemaMode | None = "auto",
    ensure_seed: bool = True,
    fail_if_schema_missing: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_action": "skipped",
        "schema_auto_initialized": False,
        "seed_enabled": ensure_seed and _bootstrap_enabled(),
        "seed_summary": None,
    }

    initial_schema_action = await apply_schema(schema_mode)
    schema_ready, summary["schema_action"], summary["schema_auto_initialized"] = await _resolve_schema_readiness(
        initial_schema_action=initial_schema_action
    )

    if not schema_ready:
        message = (
            "Core database tables are missing. "
            "Enable DATABASE_AUTO_MIGRATE or DATABASE_AUTO_CREATE, "
            "set DATABASE_AUTO_INIT_ON_STARTUP=true, "
            "or run scripts/init_project.py first."
        )
        if fail_if_schema_missing:
            raise RuntimeError(message)
        logger.warning(message)
        return summary

    if ensure_seed:
        summary["seed_summary"] = await ensure_seed_data()

    return summary


async def _create_all_schema() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


def _build_alembic_config() -> Config:
    config = Config(str(settings.root_dir / "alembic.ini"))
    config.set_main_option("script_location", str(settings.root_dir / "alembic"))

    database_url = settings.resolved_database_string
    if not database_url:
        raise RuntimeError(
            "Database migration requires DATABASE_STRING or a convertible DATABASE_ASYNC_STRING."
        )

    settings.database_string = database_url
    os.environ["DATABASE_STRING"] = database_url
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _run_migrations_sync() -> None:
    command.upgrade(_build_alembic_config(), "head")


async def _run_migrations() -> None:
    await asyncio.to_thread(_run_migrations_sync)


async def get_existing_table_names() -> set[str]:
    async with async_engine.begin() as conn:
        return set(await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names()))


async def has_core_schema() -> bool:
    required_tables = set(SQLModel.metadata.tables.keys())
    existing_tables = await get_existing_table_names()
    return required_tables.issubset(existing_tables)


def _resolve_schema_mode(mode: SchemaMode | None) -> SchemaMode | None:
    if mode != "auto":
        return mode
    if settings.resolved_database_string:
        return "migrate"
    return "create_all"


async def apply_schema(mode: SchemaMode | None) -> str:
    resolved_mode = _resolve_schema_mode(mode)
    if resolved_mode is None:
        return "skipped"
    if resolved_mode == "migrate":
        await _run_migrations()
        logger.info("Database schema initialized via Alembic migrations.")
        return "migrated"
    if resolved_mode == "create_all":
        await _create_all_schema()
        logger.info("Database schema initialized via SQLModel metadata.create_all().")
        return "created"
    raise RuntimeError(f"Unsupported schema mode: {resolved_mode}")


async def _upsert_department(
    session,
    seed: SeedDepartment,
) -> tuple[int, str]:
    result = await session.execute(
        select(DepartmentModel).where(DepartmentModel.dept_id == seed.dept_id)
    )
    department = result.scalar_one_or_none()

    if department is None:
        result = await session.execute(
            select(DepartmentModel).where(DepartmentModel.dept_name == seed.dept_name)
        )
        department = result.scalar_one_or_none()

    if department is None:
        department = DepartmentModel(dept_id=seed.dept_id, dept_name=seed.dept_name)
        session.add(department)
        await session.flush()
        return int(department.dept_id), "created"

    changed = False
    if department.dept_name != seed.dept_name:
        department.dept_name = seed.dept_name
        changed = True
    session.add(department)
    await session.flush()
    return int(department.dept_id), "updated" if changed else "kept"


async def _upsert_role(
    session,
    seed: SeedRole,
) -> tuple[int, str]:
    result = await session.execute(select(RoleModel).where(RoleModel.role_id == seed.role_id))
    role = result.scalar_one_or_none()

    if role is None:
        result = await session.execute(select(RoleModel).where(RoleModel.role_name == seed.role_name))
        role = result.scalar_one_or_none()

    if role is None:
        role = RoleModel(role_id=seed.role_id, role_name=seed.role_name)
        session.add(role)
        await session.flush()
        return int(role.role_id), "created"

    changed = False
    if role.role_name != seed.role_name:
        role.role_name = seed.role_name
        changed = True
    session.add(role)
    await session.flush()
    return int(role.role_id), "updated" if changed else "kept"


async def _ensure_role_department_mapping(
    session,
    *,
    role_id: int,
    dept_id: int,
) -> str:
    result = await session.execute(
        select(RoleDepartmentModel).where(
            RoleDepartmentModel.role_id == role_id,
            RoleDepartmentModel.dept_id == dept_id,
        )
    )
    mapping = result.scalar_one_or_none()
    if mapping is not None:
        return "kept"

    max_id_result = await session.execute(select(func.max(RoleDepartmentModel.r_d_id)))
    next_mapping_id = int(max_id_result.scalar() or 0) + 1
    session.add(
        RoleDepartmentModel(
            r_d_id=next_mapping_id,
            role_id=role_id,
            dept_id=dept_id,
        )
    )
    await session.flush()
    return "created"


async def _upsert_user(
    session,
    seed: SeedUser,
    *,
    dept_id: int,
    role_id: int,
) -> tuple[UserModel, str]:
    normalized_user_type = normalize_user_type(seed.user_type)
    result = await session.execute(select(UserModel).where(UserModel.username == seed.username))
    user = result.scalar_one_or_none()
    if user is None:
        user = UserModel(
            username=seed.username,
            password=hash_password(seed.password),
            dept_id=dept_id,
            role_id=role_id,
            user_type=normalized_user_type,
        )
        session.add(user)
        await session.flush()
        return user, "created"

    changed = False
    if seed.update_password_on_bootstrap and not verify_password(seed.password, user.password):
        user.password = hash_password(seed.password)
        changed = True
    if user.dept_id != dept_id:
        user.dept_id = dept_id
        changed = True
    if user.role_id != role_id:
        user.role_id = role_id
        changed = True
    if user.user_type != normalized_user_type:
        user.user_type = normalized_user_type
        changed = True
    session.add(user)
    await session.flush()
    return user, "updated" if changed else "kept"


def _serialize_topics(topics: list[str]) -> str:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in topics:
        topic = str(item).strip()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        normalized.append(topic)
    return json.dumps(normalized, ensure_ascii=False)


async def _ensure_user_profile(session, user_id: int, seed: SeedUser) -> str:
    result = await session.execute(select(UserProfileModel).where(UserProfileModel.user_id == user_id))
    profile = result.scalar_one_or_none()
    now = get_current_time()

    if profile is None:
        profile = UserProfileModel(
            user_id=user_id,
            answer_style=seed.answer_style,
            preferred_language=seed.preferred_language,
            preferred_topics=_serialize_topics(seed.preferred_topics),
            prefers_citations=seed.prefers_citations,
            allow_web_search=seed.allow_web_search,
            profile_notes=seed.profile_notes.strip(),
            created_time=now,
            updated_time=now,
        )
        session.add(profile)
        await session.flush()
        return "created"

    changed = False
    topics = _serialize_topics(seed.preferred_topics)
    if profile.answer_style != seed.answer_style:
        profile.answer_style = seed.answer_style
        changed = True
    if profile.preferred_language != seed.preferred_language:
        profile.preferred_language = seed.preferred_language
        changed = True
    if profile.preferred_topics != topics:
        profile.preferred_topics = topics
        changed = True
    if bool(profile.prefers_citations) != bool(seed.prefers_citations):
        profile.prefers_citations = seed.prefers_citations
        changed = True
    if bool(profile.allow_web_search) != bool(seed.allow_web_search):
        profile.allow_web_search = seed.allow_web_search
        changed = True
    notes = seed.profile_notes.strip()
    if (profile.profile_notes or "") != notes:
        profile.profile_notes = notes
        changed = True
    if changed:
        profile.updated_time = now
    session.add(profile)
    await session.flush()
    return "updated" if changed else "kept"


async def ensure_seed_data() -> dict[str, int]:
    if not _bootstrap_enabled():
        logger.info("Bootstrap seed is disabled.")
        return {
            "departments_created": 0,
            "departments_updated": 0,
            "roles_created": 0,
            "roles_updated": 0,
            "role_department_created": 0,
            "users_created": 0,
            "users_updated": 0,
            "profiles_created": 0,
            "profiles_updated": 0,
        }

    departments, roles, users = load_bootstrap_plan()

    summary = {
        "departments_created": 0,
        "departments_updated": 0,
        "roles_created": 0,
        "roles_updated": 0,
        "role_department_created": 0,
        "users_created": 0,
        "users_updated": 0,
        "profiles_created": 0,
        "profiles_updated": 0,
    }

    async with async_session_maker() as session:
        department_ref_map: dict[int, int] = {}
        role_ref_map: dict[int, int] = {}

        for seed in departments:
            actual_dept_id, action = await _upsert_department(session, seed)
            department_ref_map[seed.dept_id] = actual_dept_id
            if action == "created":
                summary["departments_created"] += 1
            elif action == "updated":
                summary["departments_updated"] += 1

        for seed in roles:
            actual_role_id, action = await _upsert_role(session, seed)
            role_ref_map[seed.role_id] = actual_role_id
            if action == "created":
                summary["roles_created"] += 1
            elif action == "updated":
                summary["roles_updated"] += 1

            for requested_dept_id in seed.dept_ids:
                actual_dept_id = department_ref_map.get(requested_dept_id, requested_dept_id)
                mapping_action = await _ensure_role_department_mapping(
                    session,
                    role_id=actual_role_id,
                    dept_id=actual_dept_id,
                )
                if mapping_action == "created":
                    summary["role_department_created"] += 1

        for seed in users:
            actual_dept_id = department_ref_map.get(seed.dept_id, seed.dept_id)
            actual_role_id = role_ref_map.get(seed.role_id, seed.role_id)

            department_exists = await session.execute(
                select(DepartmentModel.dept_id).where(DepartmentModel.dept_id == actual_dept_id)
            )
            if department_exists.scalar_one_or_none() is None:
                raise RuntimeError(
                    f"Bootstrap user '{seed.username}' references missing department id={actual_dept_id}."
                )

            role_exists = await session.execute(
                select(RoleModel.role_id).where(RoleModel.role_id == actual_role_id)
            )
            if role_exists.scalar_one_or_none() is None:
                raise RuntimeError(
                    f"Bootstrap user '{seed.username}' references missing role id={actual_role_id}."
                )

            user, action = await _upsert_user(
                session,
                seed,
                dept_id=actual_dept_id,
                role_id=actual_role_id,
            )
            if action == "created":
                summary["users_created"] += 1
            elif action == "updated":
                summary["users_updated"] += 1

            profile_action = await _ensure_user_profile(session, int(user.id), seed)
            if profile_action == "created":
                summary["profiles_created"] += 1
            elif profile_action == "updated":
                summary["profiles_updated"] += 1

        await session.commit()

    logger.warning("Bootstrap seed completed: %s", summary)
    return summary

