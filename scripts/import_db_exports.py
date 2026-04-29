from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import dotenv
import psycopg2
from bson import json_util
from psycopg2.extras import Json, execute_values
from pymongo import MongoClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.settings import settings


POSTGRES_EXPORT_DIR = ROOT_DIR / "db" / "postgre"
MONGO_EXPORT_DIR = ROOT_DIR / "db" / "mongodb"


def load_postgres_export(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if len(payload) != 1:
        raise ValueError(f"{path} should contain exactly one top-level table key")
    table_name, rows = next(iter(payload.items()))
    if not isinstance(rows, list):
        raise ValueError(f"{path} table payload is not a list")
    return table_name, rows


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def get_column_types(conn, table_name: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select column_name, coalesce(udt_name, data_type)
            from information_schema.columns
            where table_schema = 'public' and table_name = %s
            order by ordinal_position
            """,
            (table_name,),
        )
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"PostgreSQL table does not exist: {table_name}")
    return {column: udt_name for column, udt_name in rows}


def normalize_postgres_value(value: Any, column_type: str) -> Any:
    if value is None:
        return None
    if column_type in {"json", "jsonb"}:
        if isinstance(value, str):
            return Json(json.loads(value))
        return Json(value)
    return value


def reset_sequence(conn, table_name: str, primary_column: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            select pg_get_serial_sequence(%s, %s)
            """,
            (f"public.{table_name}", primary_column),
        )
        sequence = cur.fetchone()[0]
        if not sequence:
            return
        cur.execute(
            f"""
            select setval(
                %s,
                coalesce((select max({quote_ident(primary_column)}) from {quote_ident(table_name)}), 1),
                (select count(*) > 0 from {quote_ident(table_name)})
            )
            """,
            (sequence,),
        )


def import_postgres() -> dict[str, int]:
    postgres_url = settings.resolved_database_string
    if not postgres_url:
        raise RuntimeError("DATABASE_STRING or DATABASE_ASYNC_STRING is required")

    exports = [load_postgres_export(path) for path in sorted(POSTGRES_EXPORT_DIR.glob("*.json"))]
    table_names = [table_name for table_name, _ in exports]

    conn = psycopg2.connect(postgres_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "truncate table "
                    + ", ".join(quote_ident(table_name) for table_name in table_names)
                    + " restart identity cascade"
                )

            counts: dict[str, int] = {}
            for table_name, rows in exports:
                if not rows:
                    counts[table_name] = 0
                    continue

                column_types = get_column_types(conn, table_name)
                export_columns = list(rows[0].keys())
                missing = [column for column in export_columns if column not in column_types]
                if missing:
                    raise RuntimeError(f"{table_name} has missing columns in database: {missing}")

                values = [
                    tuple(normalize_postgres_value(row.get(column), column_types[column]) for column in export_columns)
                    for row in rows
                ]
                insert_sql = (
                    f"insert into {quote_ident(table_name)} "
                    f"({', '.join(quote_ident(column) for column in export_columns)}) values %s"
                )
                with conn.cursor() as cur:
                    execute_values(cur, insert_sql, values, page_size=500)

                primary_column = export_columns[0]
                reset_sequence(conn, table_name, primary_column)
                counts[table_name] = len(rows)
        return counts
    finally:
        conn.close()


def mongo_collection_from_export(path: Path) -> str:
    # Exports are named like rag_agent.rag_doc.json.
    parts = path.name.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return path.stem


def import_mongo() -> dict[str, int]:
    if not settings.mongodb_url or not settings.mongodb_db_name:
        raise RuntimeError("MONGODB_URL and MONGODB_DB_NAME are required")

    client = MongoClient(settings.mongodb_url)
    try:
        client.admin.command("ping")
        database = client[settings.mongodb_db_name]
        counts: dict[str, int] = {}
        for path in sorted(MONGO_EXPORT_DIR.glob("*.json")):
            collection_name = mongo_collection_from_export(path)
            with path.open("r", encoding="utf-8") as fh:
                rows = json_util.loads(fh.read())
            if not isinstance(rows, list):
                raise ValueError(f"{path} payload is not a JSON array")

            collection = database[collection_name]
            collection.delete_many({})
            if rows:
                collection.insert_many(rows, ordered=False)
            counts[collection_name] = len(rows)
        return counts
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Overwrite local databases with db/ JSON exports.")
    parser.add_argument("--postgres-only", action="store_true")
    parser.add_argument("--mongo-only", action="store_true")
    args = parser.parse_args()

    dotenv.load_dotenv(ROOT_DIR / ".env", override=False)

    if args.postgres_only and args.mongo_only:
        raise SystemExit("--postgres-only and --mongo-only cannot be used together")

    if not args.mongo_only:
        print("Importing PostgreSQL exports...")
        for table_name, count in import_postgres().items():
            print(f"  {table_name}: {count}")

    if not args.postgres_only:
        print("Importing MongoDB exports...")
        for collection_name, count in import_mongo().items():
            print(f"  {collection_name}: {count}")


if __name__ == "__main__":
    main()
