import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Configure cache/session behavior before importing cognee internals.
os.environ.setdefault("CACHING", "true")
os.environ.setdefault("CACHE_BACKEND", "fs")
os.environ.setdefault("AUTO_FEEDBACK", "true")
os.environ.setdefault("ENV", "dev")

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import cognee
from cognee.api.v1.search import SearchType
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
DEMO_DATA_DIR = BASE_DIR / ".data_storage"
DEMO_SYSTEM_DIR = BASE_DIR / ".cognee_system"

DATASET_NAME = "session_feedback_weights_demo"
DEFAULT_SESSION_ID = "demo_session"
MEMIFY_ALPHA = 0.619
REQUIRED_ENV_SETTINGS = {
    "CACHING": "true",
    "AUTO_FEEDBACK": "true",
    "CACHE_BACKEND": "fs",
}


class SendPayload(BaseModel):
    question: str = Field(min_length=1)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)


class FeedbackPayload(BaseModel):
    session_id: str = Field(min_length=1)
    qa_id: str = Field(min_length=1)
    feedback_score: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None


class MemifyPayload(BaseModel):
    session_id: str = Field(min_length=1)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_setting(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_config_gate_result() -> dict[str, Any]:
    current = {name: _normalize_setting(os.getenv(name)) for name in REQUIRED_ENV_SETTINGS}
    expected = dict(REQUIRED_ENV_SETTINGS)
    mismatches = [
        {
            "name": name,
            "expected": expected_value,
            "current": current.get(name, ""),
        }
        for name, expected_value in expected.items()
        if current.get(name, "") != expected_value
    ]
    return {
        "ok": len(mismatches) == 0,
        "expected": expected,
        "current": current,
        "mismatches": mismatches,
    }


def _ensure_required_settings() -> None:
    gate = _build_config_gate_result()
    if gate["ok"]:
        return
    mismatch_lines = [
        f"{entry['name']}={entry['current']!r} (expected {entry['expected']!r})"
        for entry in gate["mismatches"]
    ]
    raise HTTPException(
        status_code=412,
        detail={
            "message": "Demo blocked: required environment settings are not configured.",
            "required": gate["expected"],
            "current": gate["current"],
            "mismatches": mismatch_lines,
        },
    )


def clamp_weight(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_demo_documents() -> list[str]:
    documents = _read_json_file(DATA_DIR / "demo_documents.json")
    if not isinstance(documents, list) or not all(isinstance(item, str) for item in documents):
        raise ValueError("demo_documents.json must contain a JSON string array")
    return documents


def _get_scripted_flow() -> dict[str, Any]:
    flow = _read_json_file(DATA_DIR / "scripted_flow.json")
    if not isinstance(flow, dict):
        raise ValueError("scripted_flow.json must contain a JSON object")
    return flow


async def _latest_qa_for_session(session_id: str):
    user = await get_default_user()
    entries = await cognee.session.get_session(session_id=session_id, last_n=1, user=user)
    if not entries:
        return None
    return entries[-1]


def _extract_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_extract_answer_text(item) for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "response", "content", "message", "summary"):
            if key in value:
                extracted = _extract_answer_text(value.get(key))
                if extracted:
                    return extracted
        return ""
    for attr in ("answer", "text", "response", "content", "message"):
        if hasattr(value, attr):
            extracted = _extract_answer_text(getattr(value, attr))
            if extracted:
                return extracted
    return str(value).strip()


def _looks_like_human_text(value: str) -> bool:
    if not value:
        return False
    cleaned = value.strip()
    if len(cleaned) < 8:
        return False
    has_letter = any(char.isalpha() for char in cleaned)
    has_space = " " in cleaned
    return has_letter and has_space


def _collect_text_candidates(value: Any, out: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if _looks_like_human_text(text):
            out.append(text)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in {
                "text",
                "answer",
                "response",
                "content",
                "summary",
                "chunk_text",
                "chunk",
                "message",
            }:
                extracted = _extract_answer_text(item)
                if _looks_like_human_text(extracted):
                    out.append(extracted)
            _collect_text_candidates(item, out)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_text_candidates(item, out)
        return
    for attr in ("text", "answer", "response", "content", "summary", "message"):
        if hasattr(value, attr):
            extracted = _extract_answer_text(getattr(value, attr))
            if _looks_like_human_text(extracted):
                out.append(extracted)


async def _safe_search(question: str, session_id: str, top_k: int = 5) -> str:
    """
    Graph mode only: use GRAPH_COMPLETION.
    """
    await _ensure_dataset_context()
    search_order = [SearchType.GRAPH_COMPLETION]

    results = None
    for search_type in search_order:
        try:
            results = await cognee.search(
                query_text=question,
                query_type=search_type,
                datasets=[DATASET_NAME],
                session_id=session_id,
                top_k=max(1, min(10, int(top_k))),
            )
            if results:
                break
        except Exception:
            continue

    if not results:
        return "No answer found for this question."

    answer_text = _extract_answer_text(results)
    if answer_text:
        return answer_text

    candidates: list[str] = []
    _collect_text_candidates(results, candidates)
    if candidates:
        return max(candidates, key=len)
    return ""


async def _snapshot_graph() -> dict[str, Any]:
    await _ensure_dataset_context()
    graph_engine = await get_graph_engine()
    nodes_data, edges_data = await graph_engine.get_graph_data()

    node_ids = [str(node_id) for node_id, _ in nodes_data]
    node_weights = await graph_engine.get_node_feedback_weights(node_ids) if node_ids else {}

    edge_object_ids: list[str] = []
    for _, _, _, edge_info in edges_data:
        edge_info = edge_info or {}
        edge_object_id = edge_info.get("edge_object_id")
        if isinstance(edge_object_id, str) and edge_object_id:
            edge_object_ids.append(edge_object_id)

    edge_weights = (
        await graph_engine.get_edge_feedback_weights(edge_object_ids) if edge_object_ids else {}
    )

    nodes: list[dict[str, Any]] = []
    for node_id, node_info in nodes_data:
        info = dict(node_info or {})
        node_id_str = str(node_id)
        effective_weight = clamp_weight(
            node_weights.get(node_id_str, info.get("feedback_weight", 0.5))
        )
        nodes.append(
            {
                "id": node_id_str,
                "label": info.get("name") or node_id_str,
                "type": info.get("type", "Unknown"),
                "feedback_weight": effective_weight,
                "properties": info,
            }
        )

    edges: list[dict[str, Any]] = []
    for source, target, relation, edge_info in edges_data:
        info = dict(edge_info or {})
        edge_object_id = info.get("edge_object_id")
        effective_weight = clamp_weight(
            edge_weights.get(edge_object_id)
            if isinstance(edge_object_id, str) and edge_object_id
            else info.get("feedback_weight", 0.5)
        )
        edge_id = (
            edge_object_id
            if isinstance(edge_object_id, str) and edge_object_id
            else f"{source}->{target}:{relation}"
        )
        edges.append(
            {
                "id": edge_id,
                "source": str(source),
                "target": str(target),
                "relation": relation,
                "feedback_weight": effective_weight,
                "properties": info,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "captured_at": now_iso(),
        },
    }


def _compute_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_nodes = {node["id"]: node.get("feedback_weight", 0.5) for node in before["nodes"]}
    before_edges = {edge["id"]: edge.get("feedback_weight", 0.5) for edge in before["edges"]}

    changed_nodes = []
    for node in after["nodes"]:
        node_id = node["id"]
        after_weight = node.get("feedback_weight", 0.5)
        before_weight = before_nodes.get(node_id, 0.5)
        if abs(after_weight - before_weight) > 1e-9:
            changed_nodes.append(
                {
                    "id": node_id,
                    "before": before_weight,
                    "after": after_weight,
                    "delta": round(after_weight - before_weight, 6),
                }
            )

    changed_edges = []
    for edge in after["edges"]:
        edge_id = edge["id"]
        after_weight = edge.get("feedback_weight", 0.5)
        before_weight = before_edges.get(edge_id, 0.5)
        if abs(after_weight - before_weight) > 1e-9:
            changed_edges.append(
                {
                    "id": edge_id,
                    "before": before_weight,
                    "after": after_weight,
                    "delta": round(after_weight - before_weight, 6),
                }
            )

    return {
        "changed_nodes": changed_nodes,
        "changed_edges": changed_edges,
        "summary": {
            "changed_node_count": len(changed_nodes),
            "changed_edge_count": len(changed_edges),
        },
    }


class DemoState:
    def __init__(self):
        self.session_id = DEFAULT_SESSION_ID
        self.dataset_name = DATASET_NAME
        self.dataset_id: Optional[Any] = None
        self.dataset_owner_id: Optional[Any] = None
        self.initialized = False
        self.activity_log: list[dict[str, Any]] = []

    def log(self, event: str, detail: str, level: str = "info"):
        self.activity_log.append(
            {
                "time": now_iso(),
                "event": event,
                "detail": detail,
                "level": level,
            }
        )
        if len(self.activity_log) > 200:
            self.activity_log = self.activity_log[-200:]


async def _ensure_dataset_context() -> None:
    user = await get_default_user()

    if state.dataset_id and state.dataset_owner_id:
        await set_database_global_context_variables(state.dataset_id, state.dataset_owner_id)
        return

    datasets = await get_authorized_existing_datasets(
        user=user,
        datasets=[state.dataset_name],
        permission_type="read",
    )
    if not datasets:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{state.dataset_name}' not found for current user context",
        )

    state.dataset_id = datasets[0].id
    state.dataset_owner_id = datasets[0].owner_id
    await set_database_global_context_variables(state.dataset_id, state.dataset_owner_id)


state = DemoState()

app = FastAPI(title="Session Feedback Weights Demo API")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/demo/config_gate")
async def config_gate():
    return _build_config_gate_result()


@app.get("/demo/state")
async def get_state():
    return {
        "initialized": state.initialized,
        "session_id": state.session_id,
        "dataset_name": state.dataset_name,
        "dataset_id": str(state.dataset_id) if state.dataset_id is not None else None,
        "activity_log": state.activity_log,
        "storage": {
            "data_root": str(DEMO_DATA_DIR),
            "system_root": str(DEMO_SYSTEM_DIR),
        },
    }


@app.get("/demo/graph")
async def get_graph():
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")
    return await _snapshot_graph()


@app.get("/demo/session")
async def get_session_content(
    session_id: Optional[str] = None,
    last_n: int = Query(default=5000, ge=1, le=5000),
):
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    target_session_id = session_id or state.session_id or DEFAULT_SESSION_ID
    user = await get_default_user()
    entries = await cognee.session.get_session(
        session_id=target_session_id, last_n=last_n, user=user
    )

    sanitized = []
    for entry in entries:
        sanitized.append(
            {
                "qa_id": getattr(entry, "qa_id", None),
                "question": getattr(entry, "question", ""),
                "answer": _extract_answer_text(getattr(entry, "answer", "")),
                "feedback_score": getattr(entry, "feedback_score", None),
                "feedback_text": getattr(entry, "feedback_text", None),
                "time": getattr(entry, "time", None),
            }
        )

    return {
        "ok": True,
        "session_id": target_session_id,
        "entries": sanitized,
        "entry_count": len(sanitized),
    }


@app.get("/demo/ingested_documents")
async def get_ingested_documents():
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    documents = _get_demo_documents()
    return {
        "ok": True,
        "dataset_name": state.dataset_name,
        "count": len(documents),
        "documents": [
            {
                "id": index + 1,
                "text": doc,
                "char_count": len(doc),
            }
            for index, doc in enumerate(documents)
        ],
    }


@app.post("/demo/init")
async def init_demo():
    _ensure_required_settings()
    steps: list[dict[str, Any]] = []

    def record_step(name: str, detail: str):
        item = {"name": name, "detail": detail, "time": now_iso()}
        steps.append(item)
        state.log(name, detail)

    try:
        record_step("Reset", "Configuring isolated demo directories")
        DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
        DEMO_SYSTEM_DIR.mkdir(parents=True, exist_ok=True)

        cognee.config.data_root_directory(str(DEMO_DATA_DIR))
        cognee.config.system_root_directory(str(DEMO_SYSTEM_DIR))

        record_step("Reset", "Pruning demo-local data and metadata")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        record_step("Ingest", "Loading deterministic demo documents")
        documents = _get_demo_documents()
        await cognee.add(documents, dataset_name=DATASET_NAME)

        record_step("Cognify", "Building knowledge graph")
        await cognee.cognify(datasets=[DATASET_NAME])

        state.initialized = True
        state.session_id = DEFAULT_SESSION_ID
        state.dataset_name = DATASET_NAME
        state.dataset_id = None
        state.dataset_owner_id = None
        await _ensure_dataset_context()

        graph = await _snapshot_graph()
        record_step("Graph Ready", f"Graph contains {graph['stats']['node_count']} nodes")

        return {
            "ok": True,
            "steps": steps,
            "session_id": state.session_id,
            "dataset_name": state.dataset_name,
            "graph": graph,
            "activity_log": state.activity_log,
        }
    except Exception as error:
        state.log("Error", f"Initialization failed: {error}", level="error")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/demo/send")
async def send_question(payload: SendPayload):
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    session_id = payload.session_id or state.session_id or DEFAULT_SESSION_ID
    state.session_id = session_id

    await _ensure_dataset_context()
    latest_before = await _latest_qa_for_session(session_id)
    qa_id_before = getattr(latest_before, "qa_id", None) if latest_before else None
    feedback_score_before = (
        getattr(latest_before, "feedback_score", None) if latest_before else None
    )
    feedback_text_before = getattr(latest_before, "feedback_text", None) if latest_before else None

    state.log("Question", f"{payload.question} [Graph-based retrieval, top_k={payload.top_k}]")
    answer = await _safe_search(payload.question, session_id, payload.top_k)
    latest_after = await _latest_qa_for_session(session_id)

    qa_id_after = getattr(latest_after, "qa_id", None) if latest_after else None
    created_new_entry = bool(qa_id_after and qa_id_after != qa_id_before)
    updated_feedback_on_same_entry = bool(
        latest_after
        and qa_id_after
        and qa_id_after == qa_id_before
        and (
            getattr(latest_after, "feedback_score", None) != feedback_score_before
            or getattr(latest_after, "feedback_text", None) != feedback_text_before
        )
    )

    qa_id = qa_id_after if (created_new_entry or updated_feedback_on_same_entry) else None
    feedback_score = (
        getattr(latest_after, "feedback_score", None)
        if (created_new_entry or updated_feedback_on_same_entry)
        else None
    )
    feedback_text = (
        getattr(latest_after, "feedback_text", None)
        if (created_new_entry or updated_feedback_on_same_entry)
        else None
    )
    if created_new_entry and latest_after is not None:
        latest_answer = _extract_answer_text(getattr(latest_after, "answer", ""))
        if latest_answer:
            answer = latest_answer
    if not answer:
        answer = "No answer text available for this result."

    if feedback_score is not None or feedback_text:
        state.log(
            "Auto Feedback",
            f"Auto feedback detected on latest entry (score={feedback_score}, text={feedback_text!r})",
        )

    state.log("Answer", answer)

    return {
        "ok": True,
        "session_id": session_id,
        "answer": answer,
        "qa_id": qa_id,
        "auto_feedback": {
            "feedback_score": feedback_score,
            "feedback_text": feedback_text,
        },
        "activity_log": state.activity_log,
    }


@app.post("/demo/feedback")
async def add_feedback(payload: FeedbackPayload):
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    user = await get_default_user()
    state.session_id = payload.session_id
    await _ensure_dataset_context()
    ok = await cognee.session.add_feedback(
        session_id=payload.session_id,
        qa_id=payload.qa_id,
        feedback_text=payload.feedback_text,
        feedback_score=payload.feedback_score,
        user=user,
    )
    if not ok:
        raise HTTPException(
            status_code=404, detail="QA entry not found or feedback could not be saved"
        )

    state.log(
        "Manual Feedback",
        f"Added score={payload.feedback_score} to qa_id={payload.qa_id}",
    )
    return {"ok": True, "activity_log": state.activity_log}


@app.post("/demo/run_memify_pipeline")
async def run_memify(payload: MemifyPayload):
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    user = await get_default_user()
    state.session_id = payload.session_id
    await _ensure_dataset_context()
    before = await _snapshot_graph()

    result = await apply_feedback_weights_pipeline(
        user=user,
        session_ids=[payload.session_id],
        dataset=DATASET_NAME,
        alpha=MEMIFY_ALPHA,
        batch_size=100,
        run_in_background=False,
    )

    after = await _snapshot_graph()
    deltas = _compute_deltas(before, after)

    state.log(
        "Memify",
        (
            "Feedback memify pipeline completed "
            f"(nodes changed={deltas['summary']['changed_node_count']}, "
            f"edges changed={deltas['summary']['changed_edge_count']})"
        ),
    )

    return {
        "ok": True,
        "session_id": payload.session_id,
        "result": result,
        "before": before,
        "after": after,
        "deltas": deltas,
        "activity_log": state.activity_log,
    }


@app.post("/demo/run_demo")
async def run_demo():
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    scripted_flow = _get_scripted_flow()
    session_id = scripted_flow.get("session_id") or DEFAULT_SESSION_ID
    questions = scripted_flow.get("questions") or []

    if not isinstance(questions, list) or len(questions) < 2:
        raise HTTPException(
            status_code=500, detail="scripted_flow.json must contain at least 2 questions"
        )

    state.session_id = session_id
    user = await get_default_user()
    await _ensure_dataset_context()

    before = await _snapshot_graph()
    turn_results = []

    for index, turn in enumerate(questions, start=1):
        question = str(turn.get("question", "")).strip()
        if not question:
            continue

        answer = await _safe_search(question, session_id)
        latest_qa = await _latest_qa_for_session(session_id)
        if latest_qa is None:
            raise HTTPException(status_code=500, detail="No QA entry found after scripted search")
        latest_answer = _extract_answer_text(getattr(latest_qa, "answer", ""))
        if latest_answer:
            answer = latest_answer
        if not answer:
            answer = "No answer text available for this result."

        qa_id = getattr(latest_qa, "qa_id", None)
        feedback_score = int(turn.get("feedback_score", 3))
        feedback_text = str(turn.get("feedback_text", "Scripted demo feedback"))

        if qa_id is not None:
            await cognee.session.add_feedback(
                session_id=session_id,
                qa_id=qa_id,
                feedback_text=feedback_text,
                feedback_score=feedback_score,
                user=user,
            )

        state.log(
            f"Run Demo Turn {index}",
            f"Q: {question} | feedback_score={feedback_score}",
        )

        turn_results.append(
            {
                "question": question,
                "answer": answer,
                "qa_id": qa_id,
                "feedback_score": feedback_score,
                "feedback_text": feedback_text,
            }
        )

    memify_result = await apply_feedback_weights_pipeline(
        user=user,
        session_ids=[session_id],
        dataset=DATASET_NAME,
        alpha=MEMIFY_ALPHA,
        batch_size=100,
        run_in_background=False,
    )

    after = await _snapshot_graph()
    deltas = _compute_deltas(before, after)

    state.log(
        "Run Demo",
        (
            "Scripted lifecycle completed "
            f"(nodes changed={deltas['summary']['changed_node_count']}, "
            f"edges changed={deltas['summary']['changed_edge_count']})"
        ),
    )

    return {
        "ok": True,
        "session_id": session_id,
        "turns": turn_results,
        "memify_result": memify_result,
        "before": before,
        "after": after,
        "deltas": deltas,
        "activity_log": state.activity_log,
    }


@app.get("/demo/scripted_flow")
async def get_scripted_flow():
    if not state.initialized:
        raise HTTPException(status_code=409, detail="Demo not initialized. Run /demo/init first.")

    flow = _get_scripted_flow()
    session_id = str(flow.get("session_id") or DEFAULT_SESSION_ID)
    questions = flow.get("questions") or []
    if not isinstance(questions, list):
        raise HTTPException(
            status_code=500, detail="Invalid scripted_flow.json: questions must be a list"
        )

    sanitized_questions: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        sanitized_questions.append(
            {
                "question": question,
                "feedback_score": int(item.get("feedback_score", 3)),
                "feedback_text": str(item.get("feedback_text", "Scripted demo feedback")),
            }
        )

    return {"ok": True, "session_id": session_id, "questions": sanitized_questions}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False)
