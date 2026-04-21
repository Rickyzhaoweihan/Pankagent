#!/usr/bin/env python3
"""
PlannerAgent API Server
FastAPI server for serving PlannerAgent queries via HTTP.

Endpoints
---------
  GET    /              — API info
  GET    /health        — health check

  Multi-turn dialogue (session-based):
  POST   /chat/start    — start a chat session (first question runs full pipeline)
  POST   /chat/message  — follow-up; smart-routed to pipeline or context-only
  GET    /chat/history  — retrieve conversation history
  DELETE /chat/end      — end session and free memory

  Interactive plan review (session-based):
  POST   /plan/start    — start interactive plan session (returns plan for review)
  POST   /plan/revise   — revise plan with a user prompt
  POST   /plan/confirm  — confirm plan and run final format/reasoning pipeline
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dataclasses import dataclass, field
import asyncio
import io
import json
import os
import sys
import time
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# Load .env BEFORE importing modules that read ANTHROPIC_API_KEY at import time
# (claude.py, qp_query_planner.py, PankBaseAgent/claude.py all read os.environ).
# .env lives next to server.py.
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed — fall back to whatever env the process inherited

# Alias CLAUDE_API_KEY -> ANTHROPIC_API_KEY for backward-compat with older .env files.
# All code downstream reads ANTHROPIC_API_KEY.
if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]

# Import the main chat function and the rigor-mode toggle
import main as _main_module
from main import (
    chat_one_round, extract_markdown, clean_response_json,
    format_plan_as_markdown, clean_user_question, run_plan_start,
    run_plan_revise, run_plan_confirm,
)
from utils import reset_cypher_queries
from stream_events import emit

# Default to rigor mode for the server
_main_module.RIGOR_MODE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — pre-initialise expensive singletons ONCE at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup, cleanup on shutdown"""
    logger.info("=" * 60)
    logger.info("Starting PlannerAgent API Server...")
    logger.info("=" * 60)

    try:
        start_time = time.time()

        # 1. Pre-load PankBaseAgent text2cypher agent + session
        logger.info("Pre-loading PankBaseAgent Text2Cypher agent...")
        from PankBaseAgent.utils import _get_text2cypher_agent, _get_pankbase_session
        _get_text2cypher_agent()
        _get_pankbase_session()

        # 2. Pre-load query-planner skill text2cypher agent (separate singleton)
        logger.info("Pre-loading query-planner Text2Cypher agent...")
        import os
        _qp_scripts = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "skills", "query-planner", "scripts"
        )
        if _qp_scripts not in sys.path:
            sys.path.insert(0, _qp_scripts)
        from qp_query_planner import _get_text2cypher_agent as _qp_get_agent
        _qp_get_agent()

        # 3. Pre-load HIRN literature scripts (avoids import-lock in threads)
        logger.info("Pre-loading HIRN literature scripts...")
        _hirn_skill_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hirn_publication_retrieval", "skills", "hirn-literature-retrieve",
        )
        if _hirn_skill_dir not in sys.path:
            sys.path.insert(0, _hirn_skill_dir)
        import scripts.scrape_hirn, scripts.resolve_ids       # noqa: F401
        import scripts.fetch_fulltext, scripts.chunk_text      # noqa: F401
        import scripts.search_chunks, scripts.query_expander   # noqa: F401

        # 3b. HPAP MySQL skill DISABLED — donor metadata now lives in Neo4j KG (donor nodes)
        # The MySQL database skill is no longer loaded or used.

        # 3c. Pre-load genomic coordinates skill
        logger.info("Pre-loading genomic coordinates skill...")
        _genomic_skill_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "PankBaseAgent", "text_to_sql",
        )
        if _genomic_skill_dir not in sys.path:
            sys.path.insert(0, _genomic_skill_dir)
        from src.text2sql_agent import Text2SQLAgent as _Text2SQLAgent  # noqa: F401

        # 3d. Pre-load ssGSEA client
        logger.info("Pre-loading ssGSEA client...")
        _ssgsea_skill_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "skills", "ssgsea",
        )
        if _ssgsea_skill_dir not in sys.path:
            sys.path.insert(0, _ssgsea_skill_dir)
        import ssgsea_client as _ssgsea_client  # noqa: F401

        # 4. Load experience buffer (optional — graceful failure)
        try:
            logger.info("Loading experience buffer...")
            from PankBaseAgent.experience_buffer import get_experience_buffer
            buffer = get_experience_buffer()
            loaded = buffer.load_best_examples(max_examples=50)
            stats = buffer.get_stats()
            logger.info(
                f"✓ Experience buffer: {len(loaded)} examples "
                f"from {stats['total_curated']} curated"
            )
        except Exception as e:
            logger.warning(f"Experience buffer not available: {e}")

        elapsed = time.time() - start_time
        logger.info(f"✓ All agents initialized in {elapsed:.2f}s")
        logger.info("=" * 60)
        logger.info("Server is ready to accept requests!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}", exc_info=True)
        raise

    yield  # Server runs here

    logger.info("Shutting down server...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PlannerAgent API",
    description="API for querying biomedical knowledge graph using natural language",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    message: str
    uptime_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Plan-confirmation session state
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 1800  # 30 minutes

@dataclass
class PlanSession:
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

_plan_sessions: dict[str, PlanSession] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> PlanSession:
    with _sessions_lock:
        session = _plan_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired")
    if time.time() - session.created_at > SESSION_TTL_SECONDS:
        with _sessions_lock:
            _plan_sessions.pop(session_id, None)
        raise HTTPException(status_code=410, detail=f"Session '{session_id}' has expired")
    return session


def _cleanup_expired_sessions():
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, s in _plan_sessions.items()
                   if now - s.created_at > SESSION_TTL_SECONDS]
        for sid in expired:
            del _plan_sessions[sid]


# ---------------------------------------------------------------------------
# Plan session JSONL logger
# ---------------------------------------------------------------------------

_PLAN_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_PLAN_LOG_PATH = os.path.join(_PLAN_LOG_DIR, "plan_sessions.jsonl")
_plan_log_lock = threading.Lock()


_LOG_MAX_LINE_BYTES = 500_000  # 500 KB hard cap per JSONL line


def _log_plan_event(session_id: str, event: str, data: dict) -> None:
    """Append a structured JSON line to logs/plan_sessions.jsonl.

    Guarantees:
    - Thread-safe via ``_plan_log_lock``
    - File is opened in append mode — never overwrites existing data
    - Failures are logged but **never** propagated to callers
    - Oversized entries are truncated, not dropped
    - Non-serialisable values fall back to ``str()``
    """
    try:
        safe_data = data if isinstance(data, dict) else {"raw": str(data)[:2000]}

        entry = {
            "session_id": session_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": safe_data,
        }

        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError, OverflowError):
            entry["data"] = {"_serialization_error": True, "event": event}
            line = json.dumps(entry, ensure_ascii=False, default=str)

        if len(line.encode("utf-8")) > _LOG_MAX_LINE_BYTES:
            entry["data"] = {
                "_truncated": True,
                "question": str(safe_data.get("question", ""))[:500],
                "event": event,
                "processing_time_ms": safe_data.get("processing_time_ms"),
            }
            line = json.dumps(entry, ensure_ascii=False, default=str)

        line += "\n"

        os.makedirs(_PLAN_LOG_DIR, exist_ok=True)
        with _plan_log_lock:
            with open(_PLAN_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:
        try:
            logger.warning("[plan-log] Failed to write %s for %s: %s", event, session_id, exc)
        except Exception:
            pass


class PlanStartRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    rigor: bool = Field(True, description="Use rigorous mode")
    use_literature: bool = Field(True, description="Run HIRN literature search in parallel")

class PlanReviseRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /plan/start")
    prompt: str = Field(..., description="Revision instruction from the user")

class PlanConfirmRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /plan/start")

class PlanResponse(BaseModel):
    session_id: str
    plan_markdown: str
    plan_json: dict
    use_literature: bool = False
    error: Optional[str] = None

class PlanConfirmResponse(BaseModel):
    answer: str = Field(..., description="Full pipeline JSON response")
    answer_markdown: str = Field("", description="Clean Markdown output identical to the CLI printout")
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Multi-turn chat session state
# ---------------------------------------------------------------------------

CHAT_SESSION_TTL_SECONDS = 3600  # 1 hour

@dataclass
class ChatSession:
    session_id: str
    history: list = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}, ...]
    rigor: bool = True
    use_literature: bool = True
    # Last new_query round's plan state (used when the user asks to revise the most recent plan)
    last_question: str = ""
    last_plan: dict = field(default_factory=dict)
    last_neo4j_results: list = field(default_factory=list)
    last_cypher_queries: list = field(default_factory=list)
    last_complexity: str = "simple"
    last_literature_result: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

_chat_sessions: dict[str, ChatSession] = {}
_chat_sessions_lock = threading.Lock()

# Token budget: ~50K tokens ≈ ~160K chars for history
_MAX_HISTORY_CHARS = 150_000


def _get_chat_session(session_id: str) -> ChatSession:
    with _chat_sessions_lock:
        session = _chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found or expired")
    if time.time() - session.last_active > CHAT_SESSION_TTL_SECONDS:
        with _chat_sessions_lock:
            _chat_sessions.pop(session_id, None)
        raise HTTPException(status_code=410, detail=f"Chat session '{session_id}' has expired")
    session.last_active = time.time()
    return session


def _cleanup_expired_chat_sessions():
    now = time.time()
    with _chat_sessions_lock:
        expired = [sid for sid, s in _chat_sessions.items()
                   if now - s.last_active > CHAT_SESSION_TTL_SECONDS]
        for sid in expired:
            del _chat_sessions[sid]


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep history within token budget. Drop oldest rounds but always keep the first round."""
    total = sum(len(m["content"]) for m in history)
    if total <= _MAX_HISTORY_CHARS:
        return history
    # Always keep the first Q+A pair (rounds 0-1) for context anchoring
    first_pair = history[:2] if len(history) >= 2 else history[:]
    rest = history[2:]
    # Drop oldest pairs from `rest` until we fit
    while rest and sum(len(m["content"]) for m in first_pair + rest) > _MAX_HISTORY_CHARS:
        rest = rest[2:]  # drop one Q+A pair at a time
    return first_pair + rest


def _classify_followup(history: list[dict], new_question: str) -> str:
    """Classify whether a follow-up needs new database queries or can be answered from context.

    Returns 'new_query' or 'context_only'.
    """
    import anthropic

    # Build a compact summary of what's been discussed
    turns = []
    for m in history[-6:]:  # last 3 rounds max for classification
        role = "User" if m["role"] == "user" else "Assistant"
        turns.append(f"{role}: {m['content'][:500]}")
    conversation_summary = "\n".join(turns)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=(
                "You classify follow-up questions in a biomedical Q&A system.\n"
                "Respond with EXACTLY one word: 'new_query' or 'context_only'.\n\n"
                "Reply 'new_query' if the question:\n"
                "- Asks about a NEW gene, SNP, disease, or entity not in prior answers\n"
                "- Asks for NEW data types (expression, QTL, pathways) not yet retrieved\n"
                "- Asks to run ssGSEA, search literature, or query a database\n\n"
                "Reply 'context_only' if the question:\n"
                "- Asks to rephrase, summarize, or simplify a prior answer\n"
                "- Asks a clarification about data already shown\n"
                "- Compares or contrasts information already in the conversation\n"
                "- Asks 'why' or 'what does this mean' about prior results"
            ),
            messages=[{
                "role": "user",
                "content": f"Conversation so far:\n{conversation_summary}\n\nNew question: {new_question}",
            }],
        )
        answer = response.content[0].text.strip().lower()
        if "context" in answer:
            return "context_only"
        return "new_query"
    except Exception as exc:
        logger.warning(f"Follow-up classification failed: {exc}")
        return "new_query"  # safe fallback: always run pipeline


def _answer_from_context(history: list[dict], new_question: str, rigor: bool = True) -> str:
    """Answer a follow-up question using only conversation history (no database queries)."""
    import anthropic

    messages = []
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": new_question})

    system_prompt = (
        "You are a biomedical research assistant continuing a conversation. "
        "The user's previous questions and your previous answers are in the conversation history. "
        "Answer the follow-up question using ONLY information from the prior conversation. "
        "Do NOT make up data, PubMed IDs, or statistics that weren't in your previous answers. "
        "If you cannot answer from the available context, say so clearly and suggest what new query would help."
    )
    if rigor:
        system_prompt += (
            "\nIMPORTANT: You are in rigorous evidence-only mode. "
            "Do NOT speculate or add information beyond what was explicitly retrieved."
        )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error(f"Context-only response failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {exc}")


class ChatStartRequest(BaseModel):
    question: str = Field(..., description="First question to start the conversation")
    rigor: bool = Field(True, description="Use rigorous evidence-only mode")
    use_literature: bool = Field(True, description="Include HIRN literature in the plan")

class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID from /chat/start")
    question: str = Field(..., description="Follow-up question")

class ChatReviseRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    prompt: str = Field(..., description="Revision instruction for the last plan")

class ChatResponse(BaseModel):
    session_id: str
    answer: str = Field("", description="Full pipeline JSON (only for new_query route)")
    answer_markdown: str = Field(..., description="Markdown answer")
    route: str = Field(..., description="'new_query' or 'context_only'")
    round: int = Field(..., description="Conversation round number (1-indexed)")
    plan_markdown: str = Field("", description="Plan summary for this round (new_query only)")
    plan_json: Optional[dict] = Field(None, description="Plan JSON for this round (new_query only)")
    processing_time_ms: float

class ChatHistoryResponse(BaseModel):
    session_id: str
    rounds: int
    history: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERVER_START_TIME = time.time()

# Lock to serialise requests that use module-level global state
_request_lock = threading.Lock()


def _run_query(question: str, rigor: bool = True) -> str:
    """Run the full pipeline under the request lock and return the answer string.

    Deprecated: kept for any remaining callers, but the chat endpoints now use
    ``_run_plan_pipeline`` + ``_run_confirm`` instead (plan-mode foundation).
    """
    with _request_lock:
        _main_module.RIGOR_MODE = rigor
        reset_cypher_queries()
        messages = []
        _, response = chat_one_round(messages, question)
        return response


def _run_plan_pipeline(question: str, use_literature: bool = True) -> dict:
    """Wrap run_plan_start under the request lock. Returns the full plan dict
    (plan, neo4j_results, cypher_queries, complexity, use_literature, literature_result)."""
    clean_q = clean_user_question(question)
    with _request_lock:
        return {"interpreted_question": clean_q, **run_plan_start(clean_q, use_literature=use_literature)}


def _run_plan_revise(
    original_question: str,
    current_plan: dict,
    neo4j_results: list,
    user_prompt: str,
    use_literature: bool,
    literature_result: str,
) -> dict:
    """Wrap run_plan_revise under the request lock."""
    with _request_lock:
        return run_plan_revise(
            original_question=original_question,
            current_plan=current_plan,
            neo4j_results=neo4j_results,
            user_prompt=user_prompt,
            use_literature=use_literature,
            literature_result=literature_result,
        )


def _run_confirm(
    question: str,
    neo4j_results: list,
    cypher_queries: list,
    complexity: str,
    use_literature: bool,
    literature_result: str,
    rigor: bool,
) -> str:
    """Wrap run_plan_confirm under the request lock. Returns the formatted answer JSON string."""
    with _request_lock:
        _main_module.RIGOR_MODE = rigor
        return run_plan_confirm(
            question=question,
            neo4j_results=neo4j_results,
            cypher_queries=cypher_queries,
            complexity=complexity,
            use_literature=use_literature,
            literature_result=literature_result,
            rigor=rigor,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PlannerAgent API",
        "version": "2.1.0",
        "status": "running",
        "endpoints": {
            "chat_start": "POST /chat/start — start multi-turn dialogue (runs plan + auto-confirms)",
            "chat_message": "POST /chat/message — follow-up; smart-routed (plan+confirm or context-only)",
            "chat_revise": "POST /chat/revise — revise the most recent plan in the session, auto-confirm",
            "chat_history": "GET /chat/history?session_id=X — get conversation history",
            "chat_end": "DELETE /chat/end?session_id=X — end session",
            "plan_start": "POST /plan/start — start interactive plan session (manual confirm)",
            "plan_revise": "POST /plan/revise — revise plan with user prompt",
            "plan_confirm": "POST /plan/confirm — confirm plan and get final answer",
            "health": "GET /health — health check",
            "docs": "GET /docs — interactive API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - SERVER_START_TIME
    return HealthResponse(
        status="healthy",
        message="Server is running and ready to accept requests",
        uptime_seconds=uptime,
    )


# ---------------------------------------------------------------------------
# Plan confirmation endpoints
# ---------------------------------------------------------------------------

@app.post("/plan/start", response_model=PlanResponse)
async def plan_start(request: PlanStartRequest):
    """Start an interactive plan session.

    Runs plan_query + translate_plan + execute_plan, stores the session,
    and returns the plan as Markdown for the user to review.
    """
    start_time = time.time()

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    raw_question = request.question.strip()
    rigor = request.rigor
    use_lit = request.use_literature
    logger.info(f"[/plan/start] Received (rigor={rigor}, literature={use_lit}): {raw_question[:100]}...")

    _cleanup_expired_sessions()

    question = clean_user_question(raw_question)

    def _do_plan():
        with _request_lock:
            return run_plan_start(question, use_literature=use_lit)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_plan)
    except Exception as e:
        logger.error(f"[/plan/start] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Planning failed: {e}")

    session_id = uuid.uuid4().hex[:12]
    session = PlanSession(
        session_id=session_id,
        original_question=question,
        rigor=rigor,
        current_plan=result["plan"],
        neo4j_results=result["neo4j_results"],
        cypher_queries=result["cypher_queries"],
        complexity=result["complexity"],
        use_literature=result.get("use_literature", False),
        literature_result=result.get("literature_result", ""),
    )

    plan_md = format_plan_as_markdown(
        question, result["plan"], result["neo4j_results"],
        use_literature=session.use_literature,
        literature_result=session.literature_result,
    )
    session.chat_history.append({"role": "user", "content": question})
    session.chat_history.append({"role": "assistant", "content": plan_md})

    with _sessions_lock:
        _plan_sessions[session_id] = session

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"[/plan/start] Session {session_id} created with {len(result['plan'].get('steps', []))} steps, literature={session.use_literature}")

    _log_plan_event(session_id, "plan_start", {
        "raw_question": raw_question,
        "clean_question": question,
        "plan": result["plan"],
        "plan_markdown": plan_md,
        "neo4j_results_count": len(result.get("neo4j_results", [])),
        "cypher_queries": result.get("cypher_queries", []),
        "complexity": result.get("complexity", "simple"),
        "use_literature": session.use_literature,
        "has_literature": bool(result.get("literature_result")),
        "rigor": rigor,
        "processing_time_ms": round(processing_time, 1),
    })

    return PlanResponse(
        session_id=session_id,
        plan_markdown=plan_md,
        plan_json=result["plan"],
        use_literature=session.use_literature,
    )


@app.post("/plan/revise", response_model=PlanResponse)
async def plan_revise(request: PlanReviseRequest):
    """Revise the current plan based on a user prompt.

    Re-runs plan_query + translate + execute with the revision context
    and returns the updated plan.
    """
    start_time = time.time()

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Revision prompt cannot be empty")

    session = _get_session(request.session_id)
    prompt = request.prompt.strip()
    logger.info(f"[/plan/revise] Session {session.session_id}: {prompt[:100]}...")

    session.chat_history.append({"role": "user", "content": prompt})

    def _do_revise():
        with _request_lock:
            return run_plan_revise(
                original_question=session.original_question,
                current_plan=session.current_plan,
                neo4j_results=session.neo4j_results,
                user_prompt=prompt,
                use_literature=session.use_literature,
                literature_result=session.literature_result,
            )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_revise)
    except Exception as e:
        logger.error(f"[/plan/revise] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Revision failed: {e}")

    # Always update session state — even soft errors return a valid plan
    session.current_plan = result["plan"]
    session.neo4j_results = result["neo4j_results"]
    session.cypher_queries = result["cypher_queries"]
    session.complexity = result["complexity"]
    session.use_literature = result.get("use_literature", session.use_literature)
    session.literature_result = result.get("literature_result", session.literature_result)

    error_msg = result.get("error")

    plan_md = format_plan_as_markdown(
        session.original_question, result["plan"], result["neo4j_results"],
        use_literature=session.use_literature,
        literature_result=session.literature_result,
    )
    if error_msg:
        plan_md = f"> **Note:** {error_msg}\n\n{plan_md}"

    session.chat_history.append({"role": "assistant", "content": plan_md})

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"[/plan/revise] Session {session.session_id} revised to {len(result['plan'].get('steps', []))} steps, literature={session.use_literature}")

    _log_plan_event(session.session_id, "plan_revise", {
        "question": session.original_question,
        "user_prompt": prompt,
        "plan": result["plan"],
        "plan_markdown": plan_md,
        "neo4j_results_count": len(result.get("neo4j_results", [])),
        "cypher_queries": result.get("cypher_queries", []),
        "complexity": result.get("complexity", "simple"),
        "use_literature": session.use_literature,
        "has_literature": bool(session.literature_result),
        "error": error_msg,
        "processing_time_ms": round(processing_time, 1),
    })

    return PlanResponse(
        session_id=session.session_id,
        plan_markdown=plan_md,
        plan_json=result["plan"],
        use_literature=session.use_literature,
    )


@app.post("/plan/confirm", response_model=PlanConfirmResponse)
async def plan_confirm(request: PlanConfirmRequest):
    """Confirm the current plan and execute the format/reasoning pipeline.

    Uses the already-retrieved neo4j_results from the session — no new
    database queries are executed.
    """
    session = _get_session(request.session_id)
    logger.info(f"[/plan/confirm] Session {session.session_id} confirmed, running final pipeline...")

    start_time = time.time()

    def _do_confirm():
        with _request_lock:
            return run_plan_confirm(
                question=session.original_question,
                neo4j_results=session.neo4j_results,
                cypher_queries=session.cypher_queries,
                complexity=session.complexity,
                use_literature=session.use_literature,
                literature_result=session.literature_result,
                rigor=session.rigor,
            )

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, _do_confirm)
    except Exception as e:
        logger.error(f"[/plan/confirm] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    processing_time = (time.time() - start_time) * 1000
    cleaned = clean_response_json(response)
    md = extract_markdown(response)

    # Extract cypher queries and follow-up questions from the response JSON
    _confirm_cypher = []
    _confirm_follow_ups = []
    _confirm_reasoning = ""
    try:
        _resp_data = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
        _resp_text = _resp_data.get("text", {})
        if isinstance(_resp_text, str):
            _resp_text = json.loads(_resp_text)
        _confirm_cypher = _resp_text.get("cypher", [])
        _confirm_follow_ups = _resp_text.get("follow_up_questions", [])
        _confirm_reasoning = _resp_text.get("reasoning_trace", "")
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    _log_plan_event(session.session_id, "plan_confirm", {
        "question": session.original_question,
        "plan": session.current_plan,
        "cypher_queries": session.cypher_queries,
        "final_answer_cypher": _confirm_cypher,
        "final_answer_markdown": md[:5000],
        "reasoning_trace": _confirm_reasoning[:3000] if _confirm_reasoning else "",
        "follow_up_questions": _confirm_follow_ups,
        "neo4j_results_count": len(session.neo4j_results),
        "complexity": session.complexity,
        "use_literature": session.use_literature,
        "has_literature": bool(session.literature_result),
        "rigor": session.rigor,
        "chat_history": session.chat_history,
        "processing_time_ms": round(processing_time, 1),
    })

    # Clean up the session after successful execution
    with _sessions_lock:
        _plan_sessions.pop(session.session_id, None)

    logger.info(f"[/plan/confirm] Done in {processing_time:.0f}ms")

    return PlanConfirmResponse(answer=cleaned, answer_markdown=md, processing_time_ms=processing_time)


# ---------------------------------------------------------------------------
# Multi-turn chat endpoints
# ---------------------------------------------------------------------------

@app.post("/chat/start", response_model=ChatResponse)
async def chat_start(request: ChatStartRequest):
    """Start a new multi-turn chat session.

    Each new_query round runs ``run_plan_start`` + ``run_plan_confirm`` (same
    foundation as ``/plan/*``). The last plan is stashed on the session so the
    user can call ``/chat/revise`` to refine it before asking a new question.
    """
    start_time = time.time()

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = request.question.strip()
    rigor = request.rigor
    use_lit = request.use_literature
    logger.info(f"[/chat/start] New session (rigor={rigor}, literature={use_lit}): {question[:100]}...")

    _cleanup_expired_chat_sessions()

    # 1. Plan
    loop = asyncio.get_running_loop()
    try:
        plan_result = await loop.run_in_executor(None, _run_plan_pipeline, question, use_lit)
    except Exception as e:
        logger.error(f"[/chat/start] plan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Planning failed: {e}")

    interpreted = plan_result.get("interpreted_question", question)
    plan = plan_result["plan"]
    neo4j_results = plan_result["neo4j_results"]
    cypher_queries = plan_result["cypher_queries"]
    complexity = plan_result.get("complexity", "simple")
    literature_result = plan_result.get("literature_result", "")
    plan_md = format_plan_as_markdown(
        interpreted, plan, neo4j_results,
        use_literature=use_lit, literature_result=literature_result,
    )

    # 2. Auto-confirm → final answer
    try:
        response = await loop.run_in_executor(
            None, _run_confirm,
            interpreted, neo4j_results, cypher_queries, complexity,
            use_lit, literature_result, rigor,
        )
    except Exception as e:
        logger.error(f"[/chat/start] confirm error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Confirm failed: {e}")

    cleaned = clean_response_json(response)
    md = extract_markdown(response)

    # 3. Create session and stash state so the user can revise
    session_id = uuid.uuid4().hex[:12]
    session = ChatSession(
        session_id=session_id,
        rigor=rigor,
        use_literature=use_lit,
        last_question=interpreted,
        last_plan=plan,
        last_neo4j_results=neo4j_results,
        last_cypher_queries=cypher_queries,
        last_complexity=complexity,
        last_literature_result=literature_result,
    )
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": md})

    with _chat_sessions_lock:
        _chat_sessions[session_id] = session

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"[/chat/start] Session {session_id} done in {processing_time:.0f}ms")

    _log_plan_event(f"chat_{session_id}", "chat_start", {
        "question": question,
        "interpreted_question": interpreted,
        "plan_type": plan.get("plan_type"),
        "plan_markdown": plan_md[:3000],
        "answer_markdown": md[:3000],
        "rigor": rigor,
        "use_literature": use_lit,
        "processing_time_ms": round(processing_time, 1),
    })

    return ChatResponse(
        session_id=session_id,
        answer=cleaned,
        answer_markdown=md,
        route="new_query",
        round=1,
        plan_markdown=plan_md,
        plan_json=plan,
        processing_time_ms=processing_time,
    )


@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(request: ChatMessageRequest):
    """Send a follow-up message in an existing chat session.

    Smart routing: Claude classifies whether the follow-up needs new database
    queries (plan → confirm) or can be answered from conversation context alone.
    new_query rounds always go through the plan pipeline (same as ``/plan/*``).
    """
    start_time = time.time()

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session = _get_chat_session(request.session_id)
    question = request.question.strip()
    current_round = len(session.history) // 2 + 1

    logger.info(f"[/chat/message] Session {session.session_id} round {current_round}: {question[:100]}...")

    loop = asyncio.get_running_loop()
    route = await loop.run_in_executor(None, _classify_followup, session.history, question)
    logger.info(f"[/chat/message] Route: {route}")

    plan_md = ""
    plan_json: Optional[dict] = None

    if route == "context_only":
        # Answer from conversation history — no database queries, no plan
        md = await loop.run_in_executor(
            None, _answer_from_context, session.history, question, session.rigor
        )
        cleaned = json.dumps({"to": "user", "text": {"summary": md}})
    else:
        # Run plan + confirm, exactly like /plan/start + /plan/confirm
        try:
            plan_result = await loop.run_in_executor(
                None, _run_plan_pipeline, question, session.use_literature
            )
            interpreted = plan_result.get("interpreted_question", question)
            plan = plan_result["plan"]
            neo4j_results = plan_result["neo4j_results"]
            cypher_queries = plan_result["cypher_queries"]
            complexity = plan_result.get("complexity", "simple")
            literature_result = plan_result.get("literature_result", "")
            plan_md = format_plan_as_markdown(
                interpreted, plan, neo4j_results,
                use_literature=session.use_literature, literature_result=literature_result,
            )
            response = await loop.run_in_executor(
                None, _run_confirm,
                interpreted, neo4j_results, cypher_queries, complexity,
                session.use_literature, literature_result, session.rigor,
            )
            cleaned = clean_response_json(response)
            md = extract_markdown(response)
            plan_json = plan
            # Stash so the user can /chat/revise
            session.last_question = interpreted
            session.last_plan = plan
            session.last_neo4j_results = neo4j_results
            session.last_cypher_queries = cypher_queries
            session.last_complexity = complexity
            session.last_literature_result = literature_result
        except Exception as e:
            logger.error(f"[/chat/message] plan+confirm error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": md})
    session.history = _trim_history(session.history)

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"[/chat/message] Session {session.session_id} round {current_round} done ({route}) in {processing_time:.0f}ms")

    _log_plan_event(f"chat_{session.session_id}", "chat_message", {
        "question": question,
        "route": route,
        "round": current_round,
        "plan_markdown": plan_md[:3000],
        "answer_markdown": md[:3000],
        "processing_time_ms": round(processing_time, 1),
    })

    return ChatResponse(
        session_id=session.session_id,
        answer=cleaned,
        answer_markdown=md,
        route=route,
        round=current_round,
        plan_markdown=plan_md,
        plan_json=plan_json,
        processing_time_ms=processing_time,
    )


@app.post("/chat/revise", response_model=ChatResponse)
async def chat_revise(request: ChatReviseRequest):
    """Revise the most recent new_query plan in the chat session, then auto-confirm.

    Lets users refine a plan (e.g. "also add QTL SNPs", "drop GO terms") without
    starting a new question. The revised plan is auto-confirmed and the updated
    answer replaces the last assistant turn in history.
    """
    start_time = time.time()

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Revision prompt cannot be empty")

    session = _get_chat_session(request.session_id)
    if not session.last_plan or not session.last_question:
        raise HTTPException(status_code=409, detail="No prior plan to revise in this session")

    prompt = request.prompt.strip()
    current_round = len(session.history) // 2  # revise updates the last round, not a new one
    logger.info(f"[/chat/revise] Session {session.session_id} revising round {current_round}: {prompt[:100]}...")

    loop = asyncio.get_running_loop()

    try:
        rev = await loop.run_in_executor(
            None, _run_plan_revise,
            session.last_question,
            session.last_plan,
            session.last_neo4j_results,
            prompt,
            session.use_literature,
            session.last_literature_result,
        )
    except Exception as e:
        logger.error(f"[/chat/revise] revise error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Revision failed: {e}")

    # Update session with revised plan state
    session.last_plan = rev["plan"]
    session.last_neo4j_results = rev["neo4j_results"]
    session.last_cypher_queries = rev["cypher_queries"]
    session.last_complexity = rev.get("complexity", session.last_complexity)
    session.use_literature = rev.get("use_literature", session.use_literature)
    session.last_literature_result = rev.get("literature_result", session.last_literature_result)

    plan_md = format_plan_as_markdown(
        session.last_question, session.last_plan, session.last_neo4j_results,
        use_literature=session.use_literature, literature_result=session.last_literature_result,
    )
    err = rev.get("error")
    if err:
        plan_md = f"> **Note:** {err}\n\n{plan_md}"

    # Auto-confirm
    try:
        response = await loop.run_in_executor(
            None, _run_confirm,
            session.last_question, session.last_neo4j_results, session.last_cypher_queries,
            session.last_complexity, session.use_literature, session.last_literature_result,
            session.rigor,
        )
    except Exception as e:
        logger.error(f"[/chat/revise] confirm error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Confirm failed: {e}")

    cleaned = clean_response_json(response)
    md = extract_markdown(response)

    # Replace the last assistant turn with the revised answer (don't add a new round)
    if session.history and session.history[-1].get("role") == "assistant":
        session.history[-1] = {"role": "assistant", "content": md}
    else:
        session.history.append({"role": "user", "content": f"[revision] {prompt}"})
        session.history.append({"role": "assistant", "content": md})
    session.history = _trim_history(session.history)

    processing_time = (time.time() - start_time) * 1000
    logger.info(f"[/chat/revise] Session {session.session_id} done in {processing_time:.0f}ms")

    _log_plan_event(f"chat_{session.session_id}", "chat_revise", {
        "prompt": prompt,
        "plan_markdown": plan_md[:3000],
        "answer_markdown": md[:3000],
        "processing_time_ms": round(processing_time, 1),
    })

    return ChatResponse(
        session_id=session.session_id,
        answer=cleaned,
        answer_markdown=md,
        route="new_query",
        round=current_round,
        plan_markdown=plan_md,
        plan_json=session.last_plan,
        processing_time_ms=processing_time,
    )


@app.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(session_id: str):
    """Retrieve conversation history for a chat session."""
    session = _get_chat_session(session_id)
    return ChatHistoryResponse(
        session_id=session.session_id,
        rounds=len(session.history) // 2,
        history=session.history,
    )


@app.delete("/chat/end")
async def chat_end(session_id: str):
    """End a chat session and free memory."""
    with _chat_sessions_lock:
        removed = _chat_sessions.pop(session_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    logger.info(f"[/chat/end] Session {session_id} ended ({len(removed.history) // 2} rounds)")
    return {"status": "ended", "session_id": session_id, "rounds": len(removed.history) // 2}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)},
    )


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import os

    port = 8080

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid port '{sys.argv[1]}'. Using default port {port}")
    elif "PORT" in os.environ:
        try:
            port = int(os.getenv("PORT"))
        except ValueError:
            print(f"Error: Invalid PORT env var. Using default port {port}")

    print("\n" + "=" * 60)
    print("Starting PlannerAgent API Server (Development Mode)")
    print("=" * 60)
    print(f"\nServer will be available at:")
    print(f"  - http://localhost:{port}")
    print(f"  - API docs: http://localhost:{port}/docs")
    print(f"  - Health check: http://localhost:{port}/health")
    print(f"\nEndpoints:")
    print(f"  POST   /chat/start    — start dialogue (plan → auto-confirm)")
    print(f"  POST   /chat/message  — follow-up (plan+confirm or context-only)")
    print(f"  POST   /chat/revise   — revise last plan, auto-confirm")
    print(f"  GET    /chat/history  — get conversation history")
    print(f"  DELETE /chat/end      — end chat session")
    print(f"  POST   /plan/start    — interactive plan session (manual confirm)")
    print(f"  POST   /plan/revise   — revise plan with user prompt")
    print(f"  POST   /plan/confirm  — confirm plan and get final answer")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
