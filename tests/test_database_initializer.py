import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


_TEST_ENV = {
    "APP_ENV": "development",
    "DATABASE_STRING": "postgresql://user:pass@localhost:5432/bootstrap_tests",
    "DATABASE_ASYNC_STRING": "postgresql+asyncpg://user:pass@localhost:5432/bootstrap_tests",
    "JWT_SECRET_KEY": "test-jwt-secret",
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from service import database_initializer


class BootstrapPlanTests(unittest.TestCase):
    def test_load_bootstrap_plan_uses_seed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            seed_path = Path(temp_dir) / "seed.json"
            seed_path.write_text(
                json.dumps(
                    {
                        "departments": [{"dept_id": 7, "dept_name": "Ops"}],
                        "roles": [{"role_id": 9, "role_name": "Operator", "dept_ids": [7]}],
                        "users": [
                            {
                                "username": "ops.admin",
                                "password": "Secret@123",
                                "dept_id": 7,
                                "role_id": 9,
                                "user_type": "admin",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(database_initializer.settings, "bootstrap_seed_file", str(seed_path)),
                patch.object(database_initializer.settings, "bootstrap_seed_departments_json", None),
                patch.object(database_initializer.settings, "bootstrap_seed_roles_json", None),
                patch.object(database_initializer.settings, "bootstrap_seed_users_json", None),
            ):
                departments, roles, users = database_initializer.load_bootstrap_plan()

        self.assertEqual(len(departments), 1)
        self.assertEqual(departments[0].dept_id, 7)
        self.assertEqual(departments[0].dept_name, "Ops")
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].role_id, 9)
        self.assertEqual(roles[0].dept_ids, [7])
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0].username, "ops.admin")
        self.assertEqual(users[0].user_type, "admin")

    def test_env_json_overrides_matching_seed_file_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            seed_path = Path(temp_dir) / "seed.json"
            seed_path.write_text(
                json.dumps(
                    {
                        "departments": [{"dept_id": 1, "dept_name": "CEO"}],
                        "roles": [{"role_id": 1, "role_name": "Administrator", "dept_ids": [1]}],
                        "users": [
                            {
                                "username": "admin",
                                "password": "Admin@123456",
                                "dept_id": 1,
                                "role_id": 1,
                                "user_type": "admin",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            override_roles_json = json.dumps(
                [{"role_id": 2, "role_name": "Auditor", "dept_ids": [1]}],
                ensure_ascii=False,
            )

            with (
                patch.object(database_initializer.settings, "bootstrap_seed_file", str(seed_path)),
                patch.object(database_initializer.settings, "bootstrap_seed_departments_json", None),
                patch.object(database_initializer.settings, "bootstrap_seed_roles_json", override_roles_json),
                patch.object(database_initializer.settings, "bootstrap_seed_users_json", None),
            ):
                departments, roles, users = database_initializer.load_bootstrap_plan()

        self.assertEqual(departments[0].dept_name, "CEO")
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].role_id, 2)
        self.assertEqual(roles[0].role_name, "Auditor")
        self.assertEqual(users[0].username, "admin")


class InitializeProjectDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_initializes_schema_when_missing_on_startup(self):
        seed_summary = {
            "departments_created": 1,
            "departments_updated": 0,
            "roles_created": 1,
            "roles_updated": 0,
            "role_department_created": 1,
            "users_created": 1,
            "users_updated": 0,
            "profiles_created": 1,
            "profiles_updated": 0,
        }

        apply_schema_mock = AsyncMock(side_effect=["skipped", "migrated"])
        has_core_schema_mock = AsyncMock(side_effect=[False, False, True])
        ensure_seed_mock = AsyncMock(return_value=seed_summary)

        with (
            patch.object(database_initializer.settings, "database_auto_init_on_startup", True),
            patch.object(database_initializer, "apply_schema", apply_schema_mock),
            patch.object(database_initializer, "has_core_schema", has_core_schema_mock),
            patch.object(database_initializer, "ensure_seed_data", ensure_seed_mock),
        ):
            summary = await database_initializer.initialize_project_database(
                schema_mode=None,
                ensure_seed=True,
                fail_if_schema_missing=True,
            )

        self.assertEqual(summary["schema_action"], "migrated")
        self.assertTrue(summary["schema_auto_initialized"])
        self.assertEqual(summary["seed_summary"], seed_summary)
        self.assertEqual(apply_schema_mock.await_count, 2)
        self.assertEqual(apply_schema_mock.await_args_list[0].args, (None,))
        self.assertEqual(apply_schema_mock.await_args_list[1].args, ("auto",))
        ensure_seed_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
