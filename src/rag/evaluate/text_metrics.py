import unicodedata
from collections import Counter


_ALLOWED_SYMBOLS = {".", "%", "-", "+", "/"}
_IGNORED_SYMBOLS = {
    ",",
    "，",
    "。",
    "；",
    ";",
    "：",
    ":",
    "、",
    "\"",
    "'",
    "“",
    "”",
    "‘",
    "’",
    "(",
    ")",
    "（",
    "）",
    "[",
    "]",
    "【",
    "】",
    "{",
    "}",
    "<",
    ">",
    "《",
    "》",
    "?",
    "？",
    "!",
    "！",
}


def normalize_text(text) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    chars: list[str] = []

    for char in normalized:
        if char.isspace() or char in _IGNORED_SYMBOLS:
            continue
        if char.isalnum() or char in _ALLOWED_SYMBOLS:
            chars.append(char)

    return "".join(chars)


def exact_match_score(prediction, ground_truth) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction, ground_truth) -> float:
    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truth)

    if not normalized_prediction and not normalized_ground_truth:
        return 1.0
    if not normalized_prediction or not normalized_ground_truth:
        return 0.0

    prediction_counts = Counter(normalized_prediction)
    ground_truth_counts = Counter(normalized_ground_truth)
    common = prediction_counts & ground_truth_counts
    overlap = sum(common.values())

    if overlap <= 0:
        return 0.0

    precision = overlap / len(normalized_prediction)
    recall = overlap / len(normalized_ground_truth)
    return 2 * precision * recall / (precision + recall)
