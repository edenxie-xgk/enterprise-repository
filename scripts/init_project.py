import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def resolve_seed_file_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (ROOT_DIR / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize project database schema and bootstrap seed data."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "migrate", "create_all", "none"],
        default="auto",
        help="Schema initialization mode. Default: auto.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only initialize schema, skip bootstrap seed data.",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only apply bootstrap seed data. Schema must already exist.",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        help=(
            "Optional bootstrap seed JSON file. Relative paths are resolved from the current working "
            "directory first, then the project root."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.schema_only and args.seed_only:
        raise SystemExit("--schema-only and --seed-only cannot be used together.")

    if args.seed_file:
        os.environ["BOOTSTRAP_SEED_FILE"] = str(resolve_seed_file_path(args.seed_file))

    from service.database_initializer import initialize_project_database

    schema_mode = None if args.seed_only or args.mode == "none" else args.mode
    ensure_seed = not args.schema_only

    summary = await initialize_project_database(
        schema_mode=schema_mode,
        ensure_seed=ensure_seed,
        fail_if_schema_missing=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
