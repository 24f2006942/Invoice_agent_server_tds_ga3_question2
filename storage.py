from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def get_database_url() -> str:
    value = os.getenv(
        "DATABASE_URL",
        "sqlite:///./invoice_agent.db",
    ).strip()

    if value.startswith("postgres://"):
        value = "postgresql+psycopg://" + value[len("postgres://"):]
    elif value.startswith("postgresql://"):
        value = "postgresql+psycopg://" + value[len("postgresql://"):]

    return value


DB_URL = get_database_url()
IS_SQLITE = DB_URL.startswith("sqlite:")

connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if IS_SQLITE
    else {}
)

engine = create_engine(
    DB_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)

metadata = MetaData()

tasks_table = Table(
    "a2a_tasks",
    metadata,
    Column("id", String(100), primary_key=True),
    Column("context_id", String(100), nullable=False),
    Column("principal", String(64), nullable=False, index=True),
    Column("state", String(50), nullable=False, index=True),
    Column("batch_id", String(200), nullable=False),
    Column("initial_message_id", String(200), nullable=False),
    Column("task_json", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
)

idempotency_table = Table(
    "a2a_idempotency",
    metadata,
    Column("principal", String(64), primary_key=True),
    Column("message_id", String(200), primary_key=True),
    Column("request_hash", String(64), nullable=False),
    Column("task_id", String(100), nullable=False, index=True),
    Column("created_at", String(40), nullable=False),
)

decision_cache_table = Table(
    "a2a_decision_cache",
    metadata,
    Column("cache_key", String(64), primary_key=True),
    Column("decision_json", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
)

# Create the tables automatically when the application starts.
metadata.create_all(engine)


def dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads(value: str) -> Any:
    return json.loads(value)


def principal_for_token(token: str) -> str:
    """Store a hash of the token rather than the token itself."""
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_idempotency(
    principal: str,
    message_id: str,
) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(idempotency_table).where(
                idempotency_table.c.principal == principal,
                idempotency_table.c.message_id == message_id,
            )
        ).mappings().first()

    return dict(row) if row else None


def get_owned_task(
    task_id: str,
    principal: str,
) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(tasks_table).where(
                tasks_table.c.id == task_id,
                tasks_table.c.principal == principal,
            )
        ).mappings().first()

    return dict(row) if row else None


def task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return loads(row["task_json"])


def list_owned_tasks(principal: str) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(tasks_table.c.task_json)
            .where(tasks_table.c.principal == principal)
            .order_by(tasks_table.c.created_at.desc())
        ).all()

    return [loads(row[0]) for row in rows]


def create_initial_task(
    *,
    task: dict[str, Any],
    principal: str,
    message_id: str,
    request_hash: str,
    batch_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """
    Store the task and its idempotency key together.

    Returns:
    created  - a new task was stored
    same     - an equivalent request already exists
    conflict - the message ID was reused with different content
    """
    now = utc_now_iso()

    with engine.begin() as conn:
        existing = conn.execute(
            select(idempotency_table).where(
                idempotency_table.c.principal == principal,
                idempotency_table.c.message_id == message_id,
            )
        ).mappings().first()

        if existing:
            if existing["request_hash"] != request_hash:
                return "conflict", None

            row = conn.execute(
                select(tasks_table).where(
                    tasks_table.c.id == existing["task_id"],
                    tasks_table.c.principal == principal,
                )
            ).mappings().first()

            return (
                ("same", loads(row["task_json"]))
                if row
                else ("conflict", None)
            )

        conn.execute(
            tasks_table.insert().values(
                id=task["id"],
                context_id=task["contextId"],
                principal=principal,
                state=task["status"]["state"],
                batch_id=batch_id,
                initial_message_id=message_id,
                task_json=dumps(task),
                created_at=now,
                updated_at=now,
            )
        )

        conn.execute(
            idempotency_table.insert().values(
                principal=principal,
                message_id=message_id,
                request_hash=request_hash,
                task_id=task["id"],
                created_at=now,
            )
        )

    return "created", None


def get_cached_decisions(
    cache_keys: list[str],
) -> dict[str, dict[str, Any]]:
    if not cache_keys:
        return {}

    output = {}

    with engine.connect() as conn:
        rows = conn.execute(
            select(decision_cache_table).where(
                decision_cache_table.c.cache_key.in_(cache_keys)
            )
        ).mappings().all()

    for row in rows:
        try:
            output[row["cache_key"]] = loads(row["decision_json"])
        except (TypeError, json.JSONDecodeError):
            # Ignore a broken cache entry so a later request can retry.
            continue

    return output


def save_cached_decisions(
    decisions: dict[str, dict[str, Any]],
) -> None:
    if not decisions:
        return

    now = utc_now_iso()

    with engine.begin() as conn:
        for key, decision in decisions.items():
            values = {
                "cache_key": key,
                "decision_json": dumps(decision),
                "created_at": now,
            }

            # Insert only when the key does not already exist.
            if engine.dialect.name == "postgresql":
                statement = (
                    pg_insert(decision_cache_table)
                    .values(**values)
                    .on_conflict_do_nothing(
                        index_elements=["cache_key"]
                    )
                )

            elif engine.dialect.name == "sqlite":
                statement = (
                    sqlite_insert(decision_cache_table)
                    .values(**values)
                    .on_conflict_do_nothing(
                        index_elements=["cache_key"]
                    )
                )

            else:
                exists = conn.execute(
                    select(decision_cache_table.c.cache_key).where(
                        decision_cache_table.c.cache_key == key
                    )
                ).first()

                if exists:
                    continue

                statement = decision_cache_table.insert().values(**values)

            conn.execute(statement)


def complete_task_atomic(
    *,
    task_id: str,
    principal: str,
    task: dict[str, Any],
    message_id: str,
    request_hash: str,
    expected_state: str,
) -> tuple[str, dict[str, Any] | None]:
    """
    Complete a task only if its state has not changed.

    This prevents completion and cancellation from both winning.
    """
    now = utc_now_iso()

    with engine.begin() as conn:
        existing = conn.execute(
            select(idempotency_table).where(
                idempotency_table.c.principal == principal,
                idempotency_table.c.message_id == message_id,
            )
        ).mappings().first()

        if existing:
            if existing["request_hash"] != request_hash:
                return "conflict", None

            row = conn.execute(
                select(tasks_table).where(
                    tasks_table.c.id == existing["task_id"],
                    tasks_table.c.principal == principal,
                )
            ).mappings().first()

            return (
                ("same", loads(row["task_json"]))
                if row
                else ("conflict", None)
            )

        result = conn.execute(
            update(tasks_table)
            .where(
                tasks_table.c.id == task_id,
                tasks_table.c.principal == principal,
                tasks_table.c.state == expected_state,
            )
            .values(
                state=task["status"]["state"],
                task_json=dumps(task),
                updated_at=now,
            )
        )

        if result.rowcount == 1:
            conn.execute(
                idempotency_table.insert().values(
                    principal=principal,
                    message_id=message_id,
                    request_hash=request_hash,
                    task_id=task_id,
                    created_at=now,
                )
            )

            return "updated", None

        # Another request may have canceled or completed the task first.
        row = conn.execute(
            select(tasks_table).where(
                tasks_table.c.id == task_id,
                tasks_table.c.principal == principal,
            )
        ).mappings().first()

        if not row:
            return "conflict", None

        return "race", loads(row["task_json"])


def record_replay_key(
    *,
    principal: str,
    message_id: str,
    request_hash: str,
    task_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """
    Save a new idempotency key for an exact replay of a completed task.
    The completed task itself is not changed.
    """
    now = utc_now_iso()

    with engine.begin() as conn:
        existing = conn.execute(
            select(idempotency_table).where(
                idempotency_table.c.principal == principal,
                idempotency_table.c.message_id == message_id,
            )
        ).mappings().first()

        if existing:
            if existing["request_hash"] != request_hash:
                return "conflict", None

            row = conn.execute(
                select(tasks_table).where(
                    tasks_table.c.id == existing["task_id"],
                    tasks_table.c.principal == principal,
                )
            ).mappings().first()

            return (
                ("same", loads(row["task_json"]))
                if row
                else ("conflict", None)
            )

        row = conn.execute(
            select(tasks_table).where(
                tasks_table.c.id == task_id,
                tasks_table.c.principal == principal,
            )
        ).mappings().first()

        if not row:
            return "conflict", None

        conn.execute(
            idempotency_table.insert().values(
                principal=principal,
                message_id=message_id,
                request_hash=request_hash,
                task_id=task_id,
                created_at=now,
            )
        )

        return "recorded", loads(row["task_json"])


def cancel_task_atomic(
    *,
    task_id: str,
    principal: str,
    canceled_task: dict[str, Any],
    expected_state: str,
) -> tuple[str, dict[str, Any] | None]:
    """Cancel a task only if its state has not changed."""
    now = utc_now_iso()

    with engine.begin() as conn:
        result = conn.execute(
            update(tasks_table)
            .where(
                tasks_table.c.id == task_id,
                tasks_table.c.principal == principal,
                tasks_table.c.state == expected_state,
            )
            .values(
                state=canceled_task["status"]["state"],
                task_json=dumps(canceled_task),
                updated_at=now,
            )
        )

        if result.rowcount == 1:
            return "canceled", None

        row = conn.execute(
            select(tasks_table).where(
                tasks_table.c.id == task_id,
                tasks_table.c.principal == principal,
            )
        ).mappings().first()

        if not row:
            return "not_found", None

        return "race", loads(row["task_json"])