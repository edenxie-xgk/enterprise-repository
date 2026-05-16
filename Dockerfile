FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:${PATH}"

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

RUN set -eux; \
    uv --version

COPY pyproject.toml uv.lock .python-version ./

RUN set -eux; \
    deps_success=""; \
    for attempt in 1 2 3 4 5; do \
        if uv sync --locked --no-install-project; then \
            deps_success="1"; \
            break; \
        fi; \
        echo "uv sync failed on attempt ${attempt}, retrying..." >&2; \
        sleep 5; \
    done; \
    test -n "$deps_success"; \
    rm -rf /root/.cache/uv

COPY . .

RUN sed -i 's/\r$//' /app/docker/backend-entrypoint.sh \
    && mkdir -p /app/service/public/uploads /app/logs \
    && chmod +x /app/docker/backend-entrypoint.sh

EXPOSE 1016

ENTRYPOINT ["/app/docker/backend-entrypoint.sh"]
