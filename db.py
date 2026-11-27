"""
Database connection (Neon Postgres).

If DATABASE_URL is set (in .env or the server's settings), the app reads its data from Postgres.
If not, it falls back to the CSV files in data/ (handy for offline development and tests).

Reliability:
  - one shared connection pool, so each question doesn't open a new connection
  - every connection is checked before use (Neon sleeps when idle; a dead connection is replaced)
  - a query that fails because the connection dropped is retried once on a fresh connection
"""
import os
import time
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

POOL_MIN = 1
POOL_MAX = 5
CONNECT_TIMEOUT_SECONDS = 15          # Neon can take a few seconds to wake up after being idle


def enabled():
    return bool(os.getenv("DATABASE_URL"))


@lru_cache(maxsize=1)
def pool():
    import atexit
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    shared = ConnectionPool(
        conninfo=os.environ["DATABASE_URL"],
        min_size=POOL_MIN,
        max_size=POOL_MAX,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None,
                "connect_timeout": CONNECT_TIMEOUT_SECONDS},
        check=ConnectionPool.check_connection,        # test each connection before handing it out
        open=True,
        name="dawai",
    )
    atexit.register(shared.close)                      # close connections cleanly when the app stops
    return shared


def _run(sql, params, fetch):
    import psycopg
    for attempt in (1, 2):
        try:
            with pool().connection() as conn:
                rows = conn.execute(sql, params)
                return fetch(rows)
        except psycopg.OperationalError:
            if attempt == 2:
                raise
            time.sleep(0.5)                            # connection dropped: one retry on a fresh one


def fetch_one(sql, params=()):
    return _run(sql, params, lambda cur: cur.fetchone())


def fetch_all(sql, params=()):
    return _run(sql, params, lambda cur: cur.fetchall())


def fetch_frame(sql, params=()):
    """Query result as a pandas DataFrame (for small tables kept in memory, like Kendras)."""
    import pandas as pd
    return pd.DataFrame(fetch_all(sql, params))


def to_pg_regex(pattern):
    """Python regex -> Postgres regex. The only difference we use: word boundary \\b is \\y in Postgres."""
    return pattern.replace(r"\b", r"\y")
