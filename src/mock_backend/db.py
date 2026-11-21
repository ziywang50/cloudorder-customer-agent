"""Minimal database layer for the mock backend: SQLite by default, Postgres by URL.

One connection per transaction keeps it thread-safe. Write transactions take the lock
they need before reading anything: SQLite's database write lock (BEGIN IMMEDIATE), or a
row lock on the order on Postgres (SELECT ... FOR UPDATE via `lock_order`). That makes
read-check-write sequences such as "is this refund still within the balance?" atomic.

SQL is written with :name placeholders and translated for psycopg.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_NAMED_PARAM = re.compile(r"(?<![:\w]):(\w+)")
_POSTGRES_UNIQUE_VIOLATION = "23505"


class UniqueViolation(Exception):
    """A write broke a UNIQUE or PRIMARY KEY constraint."""


def _is_unique_violation(exc: Exception) -> bool:
    if isinstance(exc, sqlite3.IntegrityError):
        return "UNIQUE constraint failed" in str(exc)
    return getattr(exc, "sqlstate", None) == _POSTGRES_UNIQUE_VIOLATION


class Transaction:
    def __init__(self, connection: Any, dialect: str) -> None:
        self._connection = connection
        self.dialect = dialect

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        if self.dialect == "postgres":
            sql = _NAMED_PARAM.sub(r"%(\1)s", sql)
        try:
            return self._connection.execute(sql, params or {})
        except Exception as exc:
            if _is_unique_violation(exc):
                raise UniqueViolation(str(exc)) from exc
            raise

    def fetchone(self, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def fetchall(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [dict(row) for row in self.execute(sql, params).fetchall()]

    def lock_order(self, order_id: str) -> None:
        """Serialize writers on one order until this transaction ends.

        Postgres locks the order row. SQLite already holds the database write lock from
        BEGIN IMMEDIATE, which is stricter.
        """
        if self.dialect == "postgres":
            self.execute("SELECT order_id FROM orders WHERE order_id = :order_id FOR UPDATE",
                         {"order_id": order_id})


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            self._target = url.removeprefix("sqlite:///")
        elif url.startswith(("postgresql://", "postgres://")):
            self.dialect = "postgres"
            self._target = url
            try:
                import psycopg  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "A Postgres URL needs psycopg: pip install -e \".[postgres]\""
                ) from exc
        else:
            raise ValueError("Database URL must start with sqlite:/// or postgresql://")

    def create_schema(self) -> None:
        script = SCHEMA_PATH.read_text(encoding="utf-8")
        if self.dialect == "sqlite":
            connection = sqlite3.connect(self._target)
            try:
                connection.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
                connection.executescript(script)
            finally:
                connection.close()
        else:
            import psycopg

            with psycopg.connect(self._target, autocommit=True) as connection:
                connection.execute(script)

    def transaction(self, *, write: bool = False):
        """Context manager yielding a Transaction that commits on success, rolls back on error."""
        if self.dialect == "sqlite":
            return self._sqlite_transaction(write)
        return self._postgres_transaction()

    @contextmanager
    def _sqlite_transaction(self, write: bool) -> Iterator[Transaction]:
        connection = sqlite3.connect(self._target, timeout=30, isolation_level=None,
                                     check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield Transaction(connection, "sqlite")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    @contextmanager
    def _postgres_transaction(self) -> Iterator[Transaction]:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self._target, autocommit=True, row_factory=dict_row) as connection:
            with connection.transaction():
                yield Transaction(connection, "postgres")
