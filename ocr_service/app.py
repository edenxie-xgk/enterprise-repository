from __future__ import annotations

import inspect
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

APP_DIR = Path(__file__).resolve().parent
PADDLEX_CACHE_DIR = APP_DIR / ".cache" / "paddlex"

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLEX_CACHE_DIR))

app = FastAPI(title="OCR Service", version="1.0.0")
logger = logging.getLogger(__name__)
RUNTIME_MODE = "lite-predict-v2"

_ocr_instances: dict[str, object] = {}
_ocr_init_lock = threading.Lock()

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _patch_paddlex_predictor_option_compatibility() -> None:
    from paddlex.inference import PaddlePredictorOption

    if getattr(PaddlePredictorOption, "_ocr_service_compat_patched", False):
        return

    signature = inspect.signature(PaddlePredictorOption.__init__)
    parameters = tuple(signature.parameters.values())
    supports_positional_model_name = len(parameters) > 1 and parameters[1].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if supports_positional_model_name:
        return

    original_init = PaddlePredictorOption.__init__

    def compat_init(self, *args, **kwargs):
        model_name = args[0] if args else None
        if len(args) > 1:
            raise TypeError(
                f"PaddlePredictorOption compatibility wrapper accepts at most one positional model_name, got {len(args)}."
            )

        original_init(self, **kwargs)
        if model_name and hasattr(self, "setdefault_by_model_name"):
            self.setdefault_by_model_name(model_name)

    PaddlePredictorOption.__init__ = compat_init
    PaddlePredictorOption._ocr_service_compat_patched = True
    logger.warning(
        "Applied PaddlePredictorOption compatibility patch for paddlex signature %s",
        signature,
    )


def _get_ocr(lang: str):
    cached = _ocr_instances.get(lang)
    if cached is not None:
        return cached

    with _ocr_init_lock:
        cached = _ocr_instances.get(lang)
        if cached is not None:
            return cached

        PADDLEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _patch_paddlex_predictor_option_compatibility()

        from paddleocr import PaddleOCR

        det_limit_side_len = max(256, _env_int("OCR_DET_LIMIT_SIDE_LEN", 960))
        instance = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=det_limit_side_len,
        )
        _ocr_instances[lang] = instance
        return instance


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    max_side = max(256, _env_int("OCR_MAX_IMAGE_SIDE", 1600))
    height, width = img.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return img

    scale = max_side / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _extract_lines_from_prediction(result: list[object], min_score: float) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []

    for page_result in result or []:
        if not hasattr(page_result, "get"):
            continue

        texts = page_result.get("rec_texts") or []
        scores = page_result.get("rec_scores") or []
        for text, score in zip(texts, scores):
            normalized_text = str(text).replace("\n", " ").strip()
            normalized_score = float(score)
            if normalized_text and normalized_score >= min_score:
                lines.append(
                    {
                        "text": normalized_text,
                        "score": normalized_score,
                    }
                )

    return lines


@app.on_event("startup")
def preload_default_ocr() -> None:
    if not _env_bool("OCR_PRELOAD_ON_STARTUP", True):
        return

    default_lang = (os.getenv("OCR_LANG") or "ch").strip() or "ch"
    logger.info("Scheduling OCR preload for lang=%s", default_lang)

    def _background_preload() -> None:
        try:
            _get_ocr(default_lang)
            logger.info("OCR preload finished for lang=%s", default_lang)
        except Exception:
            logger.exception("OCR preload failed for lang=%s; service will stay up and retry lazily.", default_lang)

    threading.Thread(
        target=_background_preload,
        name=f"ocr-preload-{default_lang}",
        daemon=True,
    ).start()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "runtime_mode": RUNTIME_MODE,
        "cache_dir": str(PADDLEX_CACHE_DIR),
        "loaded_langs": sorted(_ocr_instances.keys()),
        "ocr_max_image_side": max(256, _env_int("OCR_MAX_IMAGE_SIDE", 1600)),
        "ocr_det_limit_side_len": max(256, _env_int("OCR_DET_LIMIT_SIDE_LEN", 960)),
    }


@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    min_score: float = Form(0.0),
    lang: str = Form("ch"),
) -> dict[str, object]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file payload.")

    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported image payload.")

    ocr = _get_ocr(lang or "ch")
    try:
        result = ocr.predict(_preprocess_image(img))
    except Exception as exc:
        logger.exception("OCR inference failed for lang=%s", lang or "ch")
        raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}") from exc

    lines = _extract_lines_from_prediction(result, min_score)

    return {
        "text": "\n".join(item["text"] for item in lines),
        "lines": lines,
        "lang": lang or "ch",
    }
