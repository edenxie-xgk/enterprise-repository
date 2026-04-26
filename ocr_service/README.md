# OCR Service

This directory contains an isolated OCR microservice so the main application can keep its current dependency tree while OCR runs in a separate virtual environment.

## Why it exists

- The main application no longer installs `PaddleOCR` locally.
- `PaddleOCR 3.x` works better in a dedicated environment because it pulls `paddlex` and related packages that conflict with the main app stack.
- Running OCR in its own environment avoids that dependency collision entirely.

## Quick start

```powershell
cd ocr_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8016
```

## Main app configuration

Set these variables in the main application's `.env`:

```env
OCR_SERVICE_URL=http://127.0.0.1:8016
OCR_SERVICE_TIMEOUT_SECONDS=120
OCR_LANG=ch
OCR_MIN_SCORE=0.5
OCR_CLIENT_MAX_CONCURRENCY=1
OCR_INFERENCE_RECOVERY_RETRIES=1
OCR_MAX_IMAGE_SIDE=1600
OCR_DET_LIMIT_SIDE_LEN=960
```

The OCR microservice also supports preloading the default model during startup so the
first ingestion request does not pay the full model initialization cost:

```env
OCR_PRELOAD_ON_STARTUP=true
OCR_LANG=ch
```
