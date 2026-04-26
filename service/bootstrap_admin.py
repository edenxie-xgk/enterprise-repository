from __future__ import annotations

from service.database_initializer import ensure_seed_data


async def ensure_bootstrap_admin() -> None:
    await ensure_seed_data()
