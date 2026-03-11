"""PostgreSQL connection manager.

Wraps psycopg2 with a context manager so callers never need to manually
manage commit/rollback/close.  The connection string is read from the
DATABASE_URL environment variable (set via .env).

Usage:
    from src.data.db import get_cursor

    with get_cursor() as cur:
        cur.execute("SELECT * FROM games WHERE season = %s", (2024,))
        rows = cur.fetchall()
"""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection_string() -> str:
    """Return the DATABASE_URL from the environment.

    Raises:
        KeyError: If DATABASE_URL is not set.  Fail fast — a missing
                  connection string is a configuration error, not a
                  runtime error to be swallowed silently.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise KeyError(
            "DATABASE_URL environment variable is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return url


@contextmanager
def get_cursor(
    connection_string: str | None = None,
) -> Generator[psycopg2.extras.DictCursor, None, None]:
    """Context manager that yields a DictCursor and handles commit/rollback.

    Args:
        connection_string: Optional override. If None, reads DATABASE_URL
                           from the environment.

    Yields:
        psycopg2 DictCursor (rows accessible by column name or index).

    On success: commits the transaction.
    On exception: rolls back and re-raises.
    """
    conn_str = connection_string or get_connection_string()
    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------

def execute_sql_file(path: str, connection_string: str | None = None) -> None:
    """Execute a SQL file against the database (e.g. run a migration).

    Args:
        path: Absolute or relative path to the .sql file.
        connection_string: Optional override; falls back to DATABASE_URL.
    """
    with open(path) as f:
        sql = f.read()
    with get_cursor(connection_string) as cur:
        cur.execute(sql)
