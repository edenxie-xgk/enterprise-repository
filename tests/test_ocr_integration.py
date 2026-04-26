import os
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import httpx

_TEST_ENV = {
    "DELETE_FILE": "false",
    "DATABASE_NAME": "test_db",
    "DATABASE_STRING": "postgresql://user:pass@localhost:5432/test_db",
    "DATABASE_ASYNC_STRING": "postgresql+asyncpg://user:pass@localhost:5432/test_db",
    "VECTOR_TABLE_NAME": "vectors",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "EMBEDDING_DIM": "1536",
    "MONGODB_URL": "mongodb://localhost:27017",
    "MONGODB_DB_NAME": "test_db",
    "DOC_COLLECTION_NAME": "docs",
    "QA_COLLECTION_NAME": "qa",
    "ELASTICSEARCH_URL": "http://localhost:9200",
    "OCR_LANG": "ch",
    "OCR_MIN_SCORE": "0.5",
    "OCR_SERVICE_TIMEOUT_SECONDS": "30",
    "OCR_CLIENT_MAX_CONCURRENCY": "1",
    "OCR_INFERENCE_RECOVERY_RETRIES": "1",
    "OCR_MAX_IMAGE_SIDE": "4096",
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from core.settings import settings
from src.rag import ocr_client
from src.rag.ingestion import loader


class RemoteOCRClientTests(unittest.TestCase):
    @patch("src.rag.ocr_client.httpx.Client")
    def test_remote_ocr_image_posts_png_and_returns_text(self, mock_client_class):
        response = MagicMock()
        response.json.return_value = {"text": "hello\nworld"}
        response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = response
        mock_client_class.return_value.__enter__.return_value = mock_client

        img = np.zeros((8, 8, 3), dtype=np.uint8)

        with (
            patch.object(settings, "ocr_service_url", "http://127.0.0.1:8016"),
            patch.object(settings, "ocr_service_timeout_seconds", 12.5),
        ):
            text = ocr_client.remote_ocr_image(
                img,
                min_score=0.7,
                language="en",
            )

        self.assertEqual(text, "hello\nworld")
        call_kwargs = mock_client.post.call_args.kwargs
        self.assertEqual(call_kwargs["data"]["min_score"], "0.7")
        self.assertEqual(call_kwargs["data"]["lang"], "en")
        self.assertEqual(call_kwargs["files"]["file"][0], "ocr.png")
        self.assertEqual(call_kwargs["files"]["file"][2], "image/png")

    @patch("src.rag.ocr_client.httpx.Client")
    def test_remote_ocr_image_resizes_large_payload_before_posting(self, mock_client_class):
        response = MagicMock()
        response.json.return_value = {"text": "scaled"}
        response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = response
        mock_client_class.return_value.__enter__.return_value = mock_client

        img = np.zeros((12, 24, 3), dtype=np.uint8)

        with (
            patch.object(settings, "ocr_service_url", "http://127.0.0.1:8016"),
            patch.object(settings, "ocr_max_image_side", 10),
        ):
            text = ocr_client.remote_ocr_image(
                img,
                min_score=0.5,
                language="ch",
            )

        self.assertEqual(text, "scaled")
        encoded_payload = mock_client.post.call_args.kwargs["files"]["file"][1]
        decoded = cv2.imdecode(np.frombuffer(encoded_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(max(decoded.shape[:2]), 10)

    @patch("src.rag.ocr_client.httpx.Client")
    def test_remote_ocr_image_retries_timeout_once(self, mock_client_class):
        response = MagicMock()
        response.json.return_value = {"text": "retry ok"}
        response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.side_effect = [
            httpx.ReadTimeout("timed out"),
            response,
        ]
        mock_client_class.return_value.__enter__.return_value = mock_client

        img = np.zeros((8, 8, 3), dtype=np.uint8)

        with (
            patch.object(settings, "ocr_service_url", "http://127.0.0.1:8016"),
            patch.object(settings, "ocr_inference_recovery_retries", 1),
        ):
            text = ocr_client.remote_ocr_image(
                img,
                min_score=0.5,
                language="ch",
            )

        self.assertEqual(text, "retry ok")
        self.assertEqual(mock_client.post.call_count, 2)

    def test_remote_ocr_image_requires_service_url(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)

        with patch.object(settings, "ocr_service_url", None):
            with self.assertRaisesRegex(RuntimeError, "OCR_SERVICE_URL"):
                ocr_client.remote_ocr_image(
                    img,
                    min_score=0.5,
                    language="ch",
                )


class LoaderOCRTests(unittest.TestCase):
    @patch("src.rag.ingestion.loader.remote_ocr_image", return_value="remote text")
    def test_ocr_image_uses_remote_service(self, mock_remote_ocr_image):
        img = np.zeros((6, 6, 3), dtype=np.uint8)

        with patch.object(settings, "orc_lang", "en"):
            text = loader.ocr_image(img, min_score=0.8)

        self.assertEqual(text, "remote text")
        mock_remote_ocr_image.assert_called_once_with(
            img,
            min_score=0.8,
            language="en",
        )


if __name__ == "__main__":
    unittest.main()
