#!/bin/sh
set -eu

wait_for_endpoint() {
  host="$1"
  port="$2"
  timeout="${WAIT_FOR_TIMEOUT_SECONDS:-90}"

  python - "$host" "$port" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect((host, port))
    except OSError as exc:
        last_error = exc
        time.sleep(2)
    else:
        print(f"[backend] dependency ready: {host}:{port}")
        raise SystemExit(0)
    finally:
        try:
            sock.close()
        except OSError:
            pass

print(f"[backend] timed out waiting for {host}:{port}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

if [ -n "${WAIT_FOR_HOSTS:-}" ]; then
  old_ifs="$IFS"
  IFS=','
  for endpoint in $WAIT_FOR_HOSTS; do
    host="${endpoint%%:*}"
    port="${endpoint##*:}"
    wait_for_endpoint "$host" "$port"
  done
  IFS="$old_ifs"
fi

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "[backend] running alembic migrations"
  python -m alembic upgrade head
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

echo "[backend] starting FastAPI server"
exec uvicorn service.server:create_server --factory --host "${SERVER_HOST:-0.0.0.0}" --port "${SERVER_PORT:-1016}"
