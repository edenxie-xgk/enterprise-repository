FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PREFER_BINARY=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN set -eux; \
    update_success=""; \
    for attempt in 1 2 3 4 5; do \
        rm -rf /var/lib/apt/lists/*; \
        if apt-get -o Acquire::Retries=3 update; then \
            update_success="1"; \
            break; \
        fi; \
        echo "apt-get update failed on attempt ${attempt}, retrying..." >&2; \
        sleep 5; \
    done; \
    test -n "$update_success"; \
    install_success=""; \
    for attempt in 1 2 3 4 5; do \
        if apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
            libgl1 \
            libglib2.0-0 \
            libgomp1 \
            libsm6 \
            libxext6 \
            libxrender1; then \
            install_success="1"; \
            break; \
        fi; \
        echo "apt-get install failed on attempt ${attempt}, retrying..." >&2; \
        apt-get -o Acquire::Retries=3 install -y -f || true; \
        sleep 5; \
    done; \
    test -n "$install_success"; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python - <<'PY'
from pathlib import Path

raw = Path("requirements.txt").read_bytes()
encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gbk", "cp936")
text = None

for encoding in encodings:
    try:
        text = raw.decode(encoding)
        break
    except UnicodeDecodeError:
        continue

if text is None:
    raise SystemExit("Unable to decode requirements.txt")

filtered = [
    line
    for line in text.splitlines()
    if line.strip()
    and not line.startswith("psycopg2==")
    and not line.startswith("torch==")
]
Path("requirements.docker.txt").write_text("\n".join(filtered) + "\n", encoding="utf-8")
PY

RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 \
    && pip install --prefer-binary -r requirements.docker.txt

COPY . .

RUN mkdir -p /app/service/public/uploads /app/logs \
    && chmod +x /app/docker/backend-entrypoint.sh

EXPOSE 1016

ENTRYPOINT ["/app/docker/backend-entrypoint.sh"]
