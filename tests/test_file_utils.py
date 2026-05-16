import unittest
from unittest.mock import patch

from fastapi import HTTPException

from service.utils.file_utils import (
    build_archived_file_name,
    build_file_download_url,
    ensure_upload_is_allowed,
    get_file_state_label,
    is_file_ready,
    sanitize_filename,
)


class FileUtilsTests(unittest.TestCase):
    def test_sanitize_filename_removes_path_and_invalid_chars(self):
        safe_name = sanitize_filename(r"..\finance/Quarterly:Report?.pdf")

        self.assertEqual(safe_name, "Quarterly_Report_.pdf")

    def test_build_file_download_url_uses_private_download_route(self):
        self.assertEqual(build_file_download_url(42), "/file/files/42/download")

    def test_file_state_helpers_expose_user_facing_status(self):
        self.assertEqual(get_file_state_label("1"), "已就绪")
        self.assertEqual(get_file_state_label("4"), "失败")
        self.assertTrue(is_file_ready("1"))
        self.assertFalse(is_file_ready("2"))
        self.assertFalse(is_file_ready("4"))

    def test_build_archived_file_name_appends_timestamp(self):
        archived_name = build_archived_file_name(
            original_file_name="report.pdf",
            create_time="2026-04-16 10:20:30",
        )

        self.assertEqual(archived_name, "report_2026-04-16_10_20_30.pdf")

    def test_ensure_upload_is_allowed_rejects_unsupported_extension(self):
        with self.assertRaises(HTTPException) as exc:
            with patch("core.settings.settings.upload_allowed_extensions", ["pdf"]):
                ensure_upload_is_allowed(file_name="report.exe", file_size=1024)

        self.assertEqual(exc.exception.status_code, 400)

    def test_ensure_upload_is_allowed_rejects_oversized_file(self):
        with self.assertRaises(HTTPException) as exc:
            with patch("core.settings.settings.upload_max_size_mb", 1):
                ensure_upload_is_allowed(file_name="report.pdf", file_size=2 * 1024 * 1024)

        self.assertEqual(exc.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
