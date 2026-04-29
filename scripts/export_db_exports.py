from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import dotenv
import psycopg2
from bson import json_util
from pymongo import MongoClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.settings import settings


POSTGRES_EXPORT_DIR = ROOT_DIR / "db" / "postgre"
MONGO_EXPORT_DIR = ROOT_DIR / "db" / "mongodb"

POSTGRES_TABLES = [
    "data_rag_doc",
    "department",
    "file",
    "role",
    "role_department",
    "users",
]
MONGO_COLLECTIONS = [
    "rag_doc",
    "rag_qa",
]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_postgres_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def postgres_export_path(table_name: str, *, overwrite: bool, suffix: str) -> Path:
    if overwrite:
        existing = sorted(POSTGRES_EXPORT_DIR.glob(f"*{table_name}*.json"))
        if table_name == "role":
            existing = [path for path in existing if "role_department" not in path.name]
        if existing:
            return existing[0]
    return POSTGRES_EXPORT_DIR / f"{table_name}_{suffix}.json"


def mongo_export_path(collection_name: str, *, overwrite: bool, db_name: str, suffix: str) -> Path:
    if overwrite:
        existing = MONGO_EXPORT_DIR / f"{db_name}.{collection_name}.json"
        if existing.exists():
            return existing
    return MONGO_EXPORT_DIR / f"{db_name}.{collection_name}.{suffix}.json"


def fetch_postgres_table(conn, table_name: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'public' and table_name = %s
            order by ordinal_position
            """,
            (table_name,),
        )
        columns = [row[0] for row in cur.fetchall()]
        if not columns:
            raise RuntimeError(f"PostgreSQL table does not exist: {table_name}")

        cur.execute(f"select {', '.join(quote_ident(column) for column in columns)} from {quote_ident(table_name)}")
        rows = cur.fetchall()

    return [
        {column: normalize_postgres_value(value) for column, value in zip(columns, row)}
        for row in rows
    ]


def export_postgres(*, overwrite: bool, suffix: str) -> dict[str, tuple[Path, int]]:
    postgres_url = settings.resolved_database_string
    if not postgres_url:
        raise RuntimeError("DATABASE_STRING or DATABASE_ASYNC_STRING is required")

    POSTGRES_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(postgres_url)
    try:
        results: dict[str, tuple[Path, int]] = {}
        for table_name in POSTGRES_TABLES:
            rows = fetch_postgres_table(conn, table_name)
            path = postgres_export_path(table_name, overwrite=overwrite, suffix=suffix)
            with path.open("w", encoding="utf-8") as fh:
                json.dump({table_name: rows}, fh, ensure_ascii=False, indent=2, default=json_default)
                fh.write("\n")
            results[table_name] = (path, len(rows))
        return results
    finally:
        conn.close()


def export_mongo(*, overwrite: bool, suffix: str) -> dict[str, tuple[Path, int]]:
    if not settings.mongodb_url or not settings.mongodb_db_name:
        raise RuntimeError("MONGODB_URL and MONGODB_DB_NAME are required")

    MONGO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    client = MongoClient(settings.mongodb_url)
    try:
        client.admin.command("ping")
        database = client[settings.mongodb_db_name]
        results: dict[str, tuple[Path, int]] = {}
        for collection_name in MONGO_COLLECTIONS:
            rows = list(database[collection_name].find({}))
            path = mongo_export_path(
                collection_name,
                overwrite=overwrite,
                db_name=settings.mongodb_db_name,
                suffix=suffix,
            )
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json_util.dumps(rows, ensure_ascii=False, indent=2))
                fh.write("\n")
            results[collection_name] = (path, len(rows))
        return results
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export current database data into db/ JSON files.")
    parser.add_argument("--postgres-only", action="store_true")
    parser.add_argument("--mongo-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing db/ export files instead of writing timestamped files.",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(ROOT_DIR / ".env", override=False)

    if args.postgres_only and args.mongo_only:
        raise SystemExit("--postgres-only and --mongo-only cannot be used together")

    suffix = timestamp_suffix()

    if not args.mongo_only:
        print("Exporting PostgreSQL tables...")
        for table_name, (path, count) in export_postgres(overwrite=args.overwrite, suffix=suffix).items():
            print(f"  {table_name}: {count} -> {path}")

    if not args.postgres_only:
        print("Exporting MongoDB collections...")
        for collection_name, (path, count) in export_mongo(overwrite=args.overwrite, suffix=suffix).items():
            print(f"  {collection_name}: {count} -> {path}")


if __name__ == "__main__":
    main()
