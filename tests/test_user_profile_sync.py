import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _import_module_with_stubs():
    sys.modules.pop("service.utils.user_profile", None)

    sqlmodel_module = types.ModuleType("sqlmodel")
    sqlmodel_module.select = lambda *args, **kwargs: None

    user_profile_model_module = types.ModuleType("service.models.user_profile")

    class UserProfileModel:
        def __init__(
            self,
            *,
            user_id=1,
            answer_style="standard",
            preferred_language="zh-CN",
            preferred_topics="",
            prefers_citations=True,
            allow_web_search=False,
            profile_notes="",
        ):
            self.user_id = user_id
            self.answer_style = answer_style
            self.preferred_language = preferred_language
            self.preferred_topics = preferred_topics
            self.prefers_citations = prefers_citations
            self.allow_web_search = allow_web_search
            self.profile_notes = profile_notes

    user_profile_model_module.UserProfileModel = UserProfileModel

    users_model_module = types.ModuleType("service.models.users")

    class UserModel:
        def __init__(self, *, id=1, username="alice", dept_id=2, role_id=3):
            self.id = id
            self.username = username
            self.dept_id = dept_id
            self.role_id = role_id

    users_model_module.UserModel = UserModel

    with patch.dict(
        sys.modules,
        {
            "sqlmodel": sqlmodel_module,
            "service.models.user_profile": user_profile_model_module,
            "service.models.users": users_model_module,
        },
    ):
        module = importlib.import_module("service.utils.user_profile")

    return module, UserModel, UserProfileModel


class UserProfileSyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module, self.UserModel, self.UserProfileModel = _import_module_with_stubs()

    def test_build_profile_sync_patch_from_query_recognizes_structured_preferences(self):
        patch, summary = self.module.build_profile_sync_patch_from_query(
            user_id="1",
            session_id=None,
            query="\u4ee5\u540e\u9ed8\u8ba4\u7528\u4e2d\u6587\u56de\u7b54\uff0c\u5e76\u4e14\u56de\u7b54\u8be6\u7ec6\u4e00\u70b9",
            user_profile={},
        )

        self.assertEqual(
            patch,
            {
                "answer_style": "detailed",
                "preferred_language": "zh-CN",
            },
        )
        self.assertIsNotNone(summary)
        self.assertEqual(summary["recognized_fields"], ["answer_style", "preferred_language"])
        self.assertIn("\u9ed8\u8ba4\u4f7f\u7528\u4e2d\u6587\u56de\u7b54", summary["candidate_summaries"])

    async def test_sync_user_profile_from_query_updates_profile_and_payload(self):
        current_user = self.UserModel()
        profile = self.UserProfileModel(preferred_language="en")

        async def fake_update_user_profile(*, session, current_user, **kwargs):
            return self.UserProfileModel(
                user_id=current_user.id,
                answer_style=kwargs.get("answer_style") or profile.answer_style,
                preferred_language=kwargs.get("preferred_language") or profile.preferred_language,
                preferred_topics=profile.preferred_topics,
                prefers_citations=profile.prefers_citations,
                allow_web_search=kwargs.get("allow_web_search")
                if kwargs.get("allow_web_search") is not None
                else profile.allow_web_search,
                profile_notes=profile.profile_notes,
            )

        with patch.object(self.module, "update_user_profile", side_effect=fake_update_user_profile):
            synced_profile, synced_payload, summary = await self.module.sync_user_profile_from_query(
                session=SimpleNamespace(),
                current_user=current_user,
                allowed_department_ids=[2, 5],
                profile=profile,
                query="\u4ee5\u540e\u9ed8\u8ba4\u7528\u4e2d\u6587\u56de\u7b54\uff0c\u5e76\u4e14\u56de\u7b54\u8be6\u7ec6\u4e00\u70b9",
            )

        self.assertTrue(summary["updated"])
        self.assertEqual(summary["applied_fields"], ["answer_style", "preferred_language"])
        self.assertEqual(synced_profile.answer_style, "detailed")
        self.assertEqual(synced_payload["preferred_language"], "zh-CN")
        self.assertEqual(synced_payload["allowed_department_ids"], [2, 5])
        self.assertEqual(summary["values"]["answer_style"], "detailed")

    async def test_sync_user_profile_from_query_is_noop_for_non_profile_memory(self):
        current_user = self.UserModel()
        profile = self.UserProfileModel()
        original_payload = self.module.build_user_profile_payload(
            current_user=current_user,
            allowed_department_ids=[2],
            profile=profile,
        )

        synced_profile, synced_payload, summary = await self.module.sync_user_profile_from_query(
            session=SimpleNamespace(),
            current_user=current_user,
            allowed_department_ids=[2],
            profile=profile,
            query="\u6211\u662f\u4e00\u540d\u7a0b\u5e8f\u5458\uff0c\u8bf7\u8bb0\u4f4f\u8fd9\u4e2a\u4fe1\u606f",
        )

        self.assertIsNone(summary)
        self.assertEqual(self.module.profile_model_to_dict(synced_profile), self.module.profile_model_to_dict(profile))
        self.assertEqual(synced_payload, original_payload)


if __name__ == "__main__":
    unittest.main()
