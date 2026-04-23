"""Tests for session_store.py.

Uses a per-test ``tmp_path`` so tests run hermetically.

We mock ``server.PlanSession`` / ``server.ChatSession`` with minimal dataclasses
that mirror the real field layout — that avoids pulling in server.py's heavy
imports (ANTHROPIC_API_KEY, vLLM client, etc.) just to test the persistence
layer.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# Fake ``server`` module with dataclasses matching server.py's definitions
# ---------------------------------------------------------------------------

@dataclass
class _FakePlanSession:
    session_id: str
    original_question: str
    rigor: bool
    chat_history: list = field(default_factory=list)
    current_plan: dict = field(default_factory=dict)
    neo4j_results: list = field(default_factory=list)
    cypher_queries: list = field(default_factory=list)
    complexity: str = "simple"
    use_literature: bool = True
    literature_result: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class _FakeChatSession:
    session_id: str
    history: list = field(default_factory=list)
    rigor: bool = True
    use_literature: bool = True
    last_question: str = ""
    last_plan: dict = field(default_factory=dict)
    last_neo4j_results: list = field(default_factory=list)
    last_cypher_queries: list = field(default_factory=list)
    last_complexity: str = "simple"
    last_literature_result: str = ""
    pending_question: str = ""
    pending_plan_session_id: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


def _install_fake_server_module():
    """Ensure ``from server import PlanSession, ChatSession`` inside
    session_store.py resolves to our fakes."""
    fake = ModuleType("server")
    fake.PlanSession = _FakePlanSession
    fake.ChatSession = _FakeChatSession
    sys.modules["server"] = fake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """Yield a fresh session_store pointed at a per-test SQLite file."""
    _install_fake_server_module()
    monkeypatch.setenv("PANK_SESSIONS_DB", str(tmp_path / "sessions.sqlite"))

    # Force re-import of session_store so it picks up the env var
    if "session_store" in sys.modules:
        del sys.modules["session_store"]
    import session_store as ss  # noqa: WPS433 — intentional re-import

    ss.init_db(str(tmp_path / "sessions.sqlite"))
    try:
        yield ss
    finally:
        ss.close_db()


def _sample_plan(**overrides) -> _FakePlanSession:
    base = _FakePlanSession(
        session_id="plan-abc123",
        original_question="Tell me about CFTR",
        rigor=True,
        chat_history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        current_plan={"plan_type": "parallel", "steps": [{"id": 1}]},
        neo4j_results=[{"query": "MATCH (g:gene) RETURN g",
                         "result": {"records": [{"g": {"name": "CFTR"}}], "keys": ["g"]}}],
        cypher_queries=["MATCH (g:gene) WHERE g.name='CFTR' RETURN g"],
        complexity="complex",
        use_literature=True,
        literature_result="x" * 50_000,
        created_at=time.time() - 10,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _sample_chat(**overrides) -> _FakeChatSession:
    base = _FakeChatSession(
        session_id="chat-xyz789",
        history=[{"role": "user", "content": "first Q"},
                 {"role": "assistant", "content": "first A"}],
        rigor=True,
        use_literature=True,
        last_question="last Q",
        last_plan={"plan_type": "parallel"},
        last_neo4j_results=[{"query": "M R R", "result": {"records": []}}],
        last_cypher_queries=["MATCH ..."],
        last_complexity="simple",
        last_literature_result="lit",
        pending_question="pending?",
        pending_plan_session_id="plan-abc123",
        created_at=time.time() - 100,
        last_active=time.time() - 5,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(store):
    with store._conn_lock:
        cur = store._get_conn().execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {row["name"] for row in cur.fetchall()}
    # Ignore sqlite internal tables like sqlite_sequence that appear once
    # an AUTOINCREMENT column is inserted into; it's not there on a fresh DB.
    assert {"plan_sessions", "chat_sessions", "events", "schema_version"} <= names


def test_init_db_idempotent(store, tmp_path):
    # Re-init with the same path: no duplicate schema_version row, no errors
    store.init_db(str(tmp_path / "sessions.sqlite"))
    assert store._count("schema_version") == 1


def test_plan_session_roundtrip(store):
    s = _sample_plan()
    store.upsert_plan_session(s)

    loaded = store.load_all_plan_sessions()
    assert len(loaded) == 1
    got = loaded[0]
    assert got.session_id == s.session_id
    assert got.original_question == s.original_question
    assert got.rigor is True
    assert got.chat_history == s.chat_history
    assert got.current_plan == s.current_plan
    assert got.neo4j_results == s.neo4j_results
    assert got.cypher_queries == s.cypher_queries
    assert got.complexity == "complex"
    assert got.use_literature is True
    assert got.literature_result == s.literature_result
    assert got.created_at == pytest.approx(s.created_at)


def test_chat_session_roundtrip(store):
    s = _sample_chat()
    store.upsert_chat_session(s)

    loaded = store.load_all_chat_sessions()
    assert len(loaded) == 1
    got = loaded[0]
    assert got.session_id == s.session_id
    assert got.history == s.history
    assert got.last_question == s.last_question
    assert got.last_plan == s.last_plan
    assert got.last_neo4j_results == s.last_neo4j_results
    assert got.pending_question == s.pending_question
    assert got.pending_plan_session_id == s.pending_plan_session_id
    assert got.created_at == pytest.approx(s.created_at)
    assert got.last_active == pytest.approx(s.last_active)


def test_upsert_updates_existing(store):
    s = _sample_plan()
    store.upsert_plan_session(s)
    time.sleep(0.01)  # ensure updated_at strictly > created_at
    s.current_plan = {"plan_type": "chain", "steps": [{"id": 1}, {"id": 2}]}
    store.upsert_plan_session(s)

    loaded = store.load_all_plan_sessions()
    assert len(loaded) == 1
    assert loaded[0].current_plan["plan_type"] == "chain"

    # Verify updated_at > created_at in the raw row
    with store._conn_lock:
        cur = store._get_conn().execute(
            "SELECT created_at, updated_at FROM plan_sessions WHERE session_id=?",
            (s.session_id,),
        )
        row = cur.fetchone()
    assert row["updated_at"] > row["created_at"]


def test_delete_plan_session(store):
    s = _sample_plan()
    store.upsert_plan_session(s)
    assert len(store.load_all_plan_sessions()) == 1

    store.delete_plan_session(s.session_id)
    assert store.load_all_plan_sessions() == []


def test_event_record(store):
    sid = "chat-abc"
    for i, ev in enumerate(["chat_start", "chat_message", "chat_end"]):
        store.record_event(sid, ev, {"round": i, "detail": f"event {i}"})

    with store._conn_lock:
        cur = store._get_conn().execute(
            "SELECT event, data FROM events WHERE session_id=? ORDER BY id ASC", (sid,)
        )
        rows = cur.fetchall()
    assert [r["event"] for r in rows] == ["chat_start", "chat_message", "chat_end"]
    assert json.loads(rows[0]["data"])["round"] == 0


def test_restart_survives_close(store, tmp_path):
    path = str(tmp_path / "sessions.sqlite")
    s = _sample_plan()
    store.upsert_plan_session(s)

    # Simulate a process restart
    store.close_db()
    store.init_db(path)

    loaded = store.load_all_plan_sessions()
    assert len(loaded) == 1
    assert loaded[0].session_id == s.session_id
    assert loaded[0].current_plan == s.current_plan


def test_malformed_json_row_skipped(store, caplog):
    # Inject a row with invalid JSON directly
    with store._conn_lock:
        store._get_conn().execute(
            """INSERT INTO plan_sessions
               (session_id, original_question, rigor, current_plan,
                neo4j_results, cypher_queries, chat_history,
                created_at, updated_at)
               VALUES (?, ?, 1, ?, '[]', '[]', '[]', ?, ?)""",
            ("plan-bad", "bad question", "not valid json{{", time.time(), time.time()),
        )

    # Add a valid row alongside
    store.upsert_plan_session(_sample_plan(session_id="plan-good"))

    with caplog.at_level("WARNING"):
        loaded = store.load_all_plan_sessions()

    ids = {s.session_id for s in loaded}
    assert ids == {"plan-good"}  # bad row skipped
    assert any("plan-bad" in rec.message for rec in caplog.records)


def test_concurrent_upserts(store):
    """8 threads × 50 upserts each → 400 distinct rows, no errors."""
    errors: list = []

    def worker(worker_id: int):
        try:
            for i in range(50):
                s = _sample_plan(session_id=f"plan-{worker_id}-{i}")
                store.upsert_plan_session(s)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store._count("plan_sessions") == 400


def test_bool_roundtrip(store):
    s = _sample_plan(rigor=False, use_literature=False)
    store.upsert_plan_session(s)
    loaded = store.load_all_plan_sessions()
    assert loaded[0].rigor is False
    assert loaded[0].use_literature is False


def test_pragma_journal_mode_wal(store):
    with store._conn_lock:
        cur = store._get_conn().execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
    assert mode.lower() == "wal"
