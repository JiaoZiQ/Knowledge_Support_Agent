import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppStorage:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_summary TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    ticket_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    data_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def ensure_session(self, user_id: str, session_id: str | None = None) -> str:
        if session_id:
            with self.connect() as connection:
                existing = connection.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
                if existing:
                    return session_id
        new_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions (id, user_id, memory_summary, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
                (new_id, user_id, now, now),
            )
        return new_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (f"msg_{uuid.uuid4().hex[:12]}", session_id, role, content, utc_now()),
            )

    def get_memory(self, session_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT memory_summary FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row["memory_summary"] if row else ""

    def update_memory(self, session_id: str, latest_user_message: str, latest_action: str) -> str:
        existing = self.get_memory(session_id)
        fragment = f"Latest user query: {latest_user_message[:80]}; latest action: {latest_action}."
        summary = (existing + " " + fragment).strip()
        if len(summary) > 420:
            summary = summary[-420:]
        with self.connect() as connection:
            connection.execute(
                "UPDATE sessions SET memory_summary = ?, updated_at = ? WHERE id = ?",
                (summary, utc_now(), session_id),
            )
        return summary

    def create_ticket(
        self,
        user_id: str,
        session_id: str | None,
        ticket_type: str,
        description: str,
        priority: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ticket_id = f"ticket_{uuid.uuid4().hex[:10]}"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tickets
                (id, user_id, session_id, ticket_type, description, priority, status, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
                """,
                (
                    ticket_id,
                    user_id,
                    session_id,
                    ticket_type,
                    description,
                    priority,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )
        return ticket_id

    def add_trace(
        self,
        session_id: str,
        user_query: str,
        action: str,
        confidence: float,
        data: dict[str, Any],
        latency_ms: float,
    ) -> str:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO traces
                (id, session_id, user_query, action, confidence, data_json, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    session_id,
                    user_query,
                    action,
                    confidence,
                    json.dumps(data, ensure_ascii=False),
                    latency_ms,
                    utc_now(),
                ),
            )
        return trace_id

    def list_tickets(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session_trace(self, session_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            session = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            messages = connection.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            traces = connection.execute(
                "SELECT * FROM traces WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return {
            "session": dict(session) if session else None,
            "messages": [dict(row) for row in messages],
            "traces": [
                {**dict(row), "data": json.loads(row["data_json"])}
                for row in traces
            ],
        }
