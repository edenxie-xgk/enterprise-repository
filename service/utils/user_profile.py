from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from service.models.user_profile import UserProfileModel
from service.models.users import UserModel
from src.memory.candidate_extractor import extract_memory_write_candidates
from src.types.memory_type import MemoryWriteCandidate, MemoryWriteRequest
from utils.utils import get_current_time


DEFAULT_PROFILE = {
    "answer_style": "standard",
    "preferred_language": "zh-CN",
    "preferred_topics": [],
    "prefers_citations": True,
    "allow_web_search": False,
    "profile_notes": "",
}


def _normalize_topics(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parsed = list(value)
    else:
        parsed = [str(value)]

    topics: list[str] = []
    seen = set()
    for item in parsed:
        topic = str(item).strip()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
    return topics[:10]


def _serialize_topics(topics: Any) -> str:
    return json.dumps(_normalize_topics(topics), ensure_ascii=False)


def _candidate_tag_value(candidate: MemoryWriteCandidate, key: str) -> str | None:
    tags = list(candidate.tags or [])
    for index, tag in enumerate(tags[:-1]):
        if str(tag).strip().lower() != key:
            continue
        value = str(tags[index + 1]).strip()
        return value or None
    return None


def profile_model_to_dict(profile: UserProfileModel | None) -> dict[str, Any]:
    if not profile:
        return dict(DEFAULT_PROFILE)
    return {
        "answer_style": profile.answer_style or DEFAULT_PROFILE["answer_style"],
        "preferred_language": profile.preferred_language or DEFAULT_PROFILE["preferred_language"],
        "preferred_topics": _normalize_topics(profile.preferred_topics),
        "prefers_citations": bool(profile.prefers_citations),
        "allow_web_search": bool(profile.allow_web_search),
        "profile_notes": profile.profile_notes or "",
    }


def build_profile_sync_patch_from_candidates(
    candidates: list[MemoryWriteCandidate],
) -> tuple[dict[str, Any], list[str], list[str]]:
    patch: dict[str, Any] = {}
    diagnostics: list[str] = []
    matched_summaries: list[str] = []
    seen_summaries: set[str] = set()

    for candidate in candidates:
        matched = False

        answer_style = _candidate_tag_value(candidate, "answer_style")
        if answer_style in {"concise", "standard", "detailed"}:
            patch["answer_style"] = answer_style
            diagnostics.append(f"profile_sync_candidate:answer_style={answer_style}")
            matched = True

        language = _candidate_tag_value(candidate, "language")
        if language:
            patch["preferred_language"] = language
            diagnostics.append(f"profile_sync_candidate:preferred_language={language}")
            matched = True

        web_search = _candidate_tag_value(candidate, "web_search")
        if web_search == "enabled":
            patch["allow_web_search"] = True
            diagnostics.append("profile_sync_candidate:allow_web_search=true")
            matched = True
        elif web_search == "disabled":
            patch["allow_web_search"] = False
            diagnostics.append("profile_sync_candidate:allow_web_search=false")
            matched = True

        summary = (candidate.summary or candidate.content or "").strip()
        if matched and summary and summary not in seen_summaries:
            seen_summaries.add(summary)
            matched_summaries.append(summary)

    return patch, diagnostics, matched_summaries


def build_profile_sync_patch_from_query(
    *,
    user_id: str,
    session_id: str | None,
    query: str,
    user_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidates = extract_memory_write_candidates(
        MemoryWriteRequest(
            user_id=user_id,
            session_id=session_id,
            query=query,
            answer="",
            chat_history=[],
            user_profile=user_profile or {},
            existing_memories=[],
        )
    )
    patch, diagnostics, matched_summaries = build_profile_sync_patch_from_candidates(candidates)
    if not patch:
        return {}, None

    return patch, {
        "recognized": True,
        "updated": False,
        "recognized_fields": sorted(patch.keys()),
        "applied_fields": [],
        "candidate_summaries": matched_summaries,
        "diagnostics": diagnostics,
    }


async def get_or_create_user_profile(
    *,
    session: AsyncSession,
    current_user: UserModel,
) -> UserProfileModel:
    result = await session.execute(
        select(UserProfileModel).where(UserProfileModel.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if profile:
        return profile

    profile = UserProfileModel(user_id=int(current_user.id))
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return profile


async def update_user_profile(
    *,
    session: AsyncSession,
    current_user: UserModel,
    answer_style: str | None = None,
    preferred_language: str | None = None,
    preferred_topics: Any = None,
    prefers_citations: bool | None = None,
    allow_web_search: bool | None = None,
    profile_notes: str | None = None,
) -> UserProfileModel:
    profile = await get_or_create_user_profile(session=session, current_user=current_user)
    if answer_style is not None:
        profile.answer_style = answer_style
    if preferred_language is not None:
        profile.preferred_language = preferred_language
    if preferred_topics is not None:
        profile.preferred_topics = _serialize_topics(preferred_topics)
    if prefers_citations is not None:
        profile.prefers_citations = prefers_citations
    if allow_web_search is not None:
        profile.allow_web_search = allow_web_search
    if profile_notes is not None:
        profile.profile_notes = profile_notes.strip()
    profile.updated_time = get_current_time()
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return profile


async def sync_user_profile_from_query(
    *,
    session: AsyncSession,
    current_user: UserModel,
    allowed_department_ids: list[int],
    profile: UserProfileModel | None,
    query: str,
) -> tuple[UserProfileModel, dict[str, Any], dict[str, Any] | None]:
    current_profile = profile or await get_or_create_user_profile(session=session, current_user=current_user)
    current_payload = build_user_profile_payload(
        current_user=current_user,
        allowed_department_ids=allowed_department_ids,
        profile=current_profile,
    )
    patch, summary = build_profile_sync_patch_from_query(
        user_id=str(current_user.id or ""),
        session_id=None,
        query=query,
        user_profile=current_payload,
    )
    if summary is None:
        return current_profile, current_payload, None

    current_profile_dict = profile_model_to_dict(current_profile)
    effective_patch = {
        key: value
        for key, value in patch.items()
        if current_profile_dict.get(key) != value
    }

    if not effective_patch:
        summary["diagnostics"].append("profile_sync_already_applied")
        return current_profile, current_payload, summary

    updated_profile = await update_user_profile(
        session=session,
        current_user=current_user,
        answer_style=effective_patch.get("answer_style"),
        preferred_language=effective_patch.get("preferred_language"),
        allow_web_search=effective_patch.get("allow_web_search"),
    )
    updated_payload = build_user_profile_payload(
        current_user=current_user,
        allowed_department_ids=allowed_department_ids,
        profile=updated_profile,
    )
    summary["updated"] = True
    summary["applied_fields"] = sorted(effective_patch.keys())
    summary["values"] = {key: updated_payload.get(key) for key in effective_patch}
    summary["diagnostics"].append("profile_sync_updated")
    return updated_profile, updated_payload, summary


def build_user_profile_payload(
    *,
    current_user: UserModel,
    allowed_department_ids: list[int],
    profile: UserProfileModel | None,
) -> dict[str, Any]:
    profile_dict = profile_model_to_dict(profile)
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "dept_id": current_user.dept_id,
        "department_id": current_user.dept_id,
        "role_id": current_user.role_id,
        "allowed_department_ids": allowed_department_ids,
        **profile_dict,
    }
