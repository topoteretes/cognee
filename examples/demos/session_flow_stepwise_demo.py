"""Step-by-step trace of the cognee 1.0 memory loop, driven by remember() + recall().

Run with:

    uv run python examples/demos/session_flow_stepwise_demo.py

This is a *narrated trace*: at every hop it prints the actual data so you can watch memory
move through the system using only the two top-level verbs, `cognee.remember` and `cognee.recall`.

It demonstrates that session DISTILLATION is now covered by remember(): distillation is wired
into improve() (see cognee/api/v1/improve/improve.py, stage 2c), and remember(session,
self_improvement=True) calls improve(session_ids=[...]) on its background bridge — so the
remembered session is curated into permanent lessons with no explicit distill_session() call.

How the pieces line up:

    remember(docs, dataset_name=...)            -> PERMANENT memory: add + cognify (+ improve)
    recall(query, session_id=...)               -> answer; with AUTO_FEEDBACK, rules/lessons
                                                   stated in a turn are absorbed as gated
                                                   "active guidance" (the distillation inputs)
    remember(text, session_id=..., self_improvement=True)
                                                -> store in session cache, then BACKGROUND
                                                   improve(session_ids=[...]) which now also
                                                   DISTILLS the gated guidance into the graph
    recall(query, session_id=<fresh>)           -> answer from PERMANENT memory only

Flow traced:

    (A)   remember(documents)              seed permanent memory
    (B+C) multi-turn recall() session      guidance absorbed each turn; every N turns
                                           improve(session_ids) auto-distills into the graph
    (D)   recall(question, fresh session)  the distilled lessons are now permanent
    (E)   visualize_graph                  distilled nodes shown with a gold ring

Requires a configured LLM provider (see CLAUDE.md). Wording of answers/lessons varies by model.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Load the repo .env BEFORE importing cognee, with override=True so the real keys in
# .env win over any stale/placeholder values exported in the shell session (pydantic
# settings otherwise prefer process env vars over the .env file).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_PATH, override=True)
except ImportError:
    # Minimal fallback parser if python-dotenv is unavailable.
    if _ENV_PATH.exists():
        for _line in _ENV_PATH.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

# litellm falls back to OPENAI_API_KEY for OpenAI calls; .env only sets LLM_API_KEY, so a
# stale/dummy OPENAI_API_KEY exported in the shell would win and auth would fail. Mirror the
# real key across unless we're running fully local via Ollama.
if os.environ.get("DEMO_USE_OLLAMA") != "1" and os.environ.get("LLM_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]

# Pin a gpt-4o-family model for the OpenAI path. cognee's current default (openai/gpt-5-mini)
# forces instructor's json_schema_mode, which calls response_model.model_json_schema() and
# therefore breaks on cognee's str-returning paths (preflight + recall answer generation):
#   AttributeError: type object 'str' has no attribute 'model_json_schema'
# gpt-4o-* uses instructor's TOOLS mode, which tolerates a primitive str response_model.
if os.environ.get("DEMO_USE_OLLAMA") != "1":
    os.environ.setdefault("LLM_MODEL", "openai/gpt-4o-mini")

# Optional fully-local run via Ollama (no API keys). Enable with DEMO_USE_OLLAMA=1.
# Applied AFTER load_dotenv(override=True) so it wins over any keys in .env.
# Requires `ollama serve` with the listed models pulled.
if os.environ.get("DEMO_USE_OLLAMA") == "1":
    os.environ.update(
        {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "llama3.1:8b",
            "LLM_ENDPOINT": "http://localhost:11434/v1",
            "LLM_API_KEY": "ollama",
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_MODEL": "nomic-embed-text:latest",
            "EMBEDDING_DIMENSIONS": "768",
            "EMBEDDING_ENDPOINT": "http://localhost:11434/api/embed",
            "HUGGINGFACE_TOKENIZER": "nomic-ai/nomic-embed-text-v1.5",
        }
    )

# Disable cognee's 30s LLM preflight via env var. The preflight calls the model with
# response_model=str, which breaks under gpt-5's instructor json_schema_mode; skipping it
# avoids that path entirely. The real pipeline calls still exercise the LLM/embeddings.
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

# AUTO_FEEDBACK makes recall() absorb rules/lessons stated in a turn into the session's
# active-guidance layer. Those gated entries are exactly what distillation curates.
os.environ["AUTO_FEEDBACK"] = "true"
os.environ.setdefault("LOG_LEVEL", "ERROR")

# --------------------------------------------------------------------------- #
# Output logging: tee every printed line to a timestamped file so the full     #
# narrated trace is captured on disk in addition to the console.               #
# --------------------------------------------------------------------------- #
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"session_flow_stepwise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
VIZ_PATH = LOG_PATH.with_suffix(".html")


class _Tee:
    """Write to the original stream and the log file at once."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()


_LOG_FILE = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _LOG_FILE)
sys.stderr = _Tee(sys.__stderr__, _LOG_FILE)

import cognee  # noqa: E402  (env must be configured before importing cognee)
from cognee import SearchType  # noqa: E402
from cognee.infrastructure.session.get_session_manager import get_session_manager  # noqa: E402
from cognee.modules.users.methods import get_default_user  # noqa: E402

DATASET_NAME = "stepwise_remember_recall_demo"
SESSION_ID = "stepwise_session"

DOCUMENTS = [
    "Aurora Robotics builds two products: the VoltaArm industrial gripper and the "
    "TerraScout warehouse rover.",
    "The VoltaArm gripper uses firmware version 4 and a calibration routine that maps "
    "joint torque to grip strength.",
    "The TerraScout rover navigates warehouses using lidar maps and charging dock beacons.",
    "Dana Voss leads the VoltaArm firmware team at Aurora Robotics.",
    "Calibration data for the VoltaArm gripper is stored in a battery-backed memory bank.",
]

# A multi-turn session: rules/lessons (learned, NOT in the seed docs) interleaved with
# questions. recall() absorbs the rules/lessons as gated guidance each turn.
SESSION_TURNS = [
    ("rule", "Always run the HALT test suite before a VoltaArm firmware release."),
    (
        "lesson",
        "Flashing VoltaArm firmware wipes calibration data, so calibration must be re-run afterwards.",
    ),
    ("question", "How does the TerraScout rover navigate warehouses?"),
    ("preference", "Keep firmware answers to a few short bullet points."),
    (
        "lesson",
        "After re-running VoltaArm calibration, verify it against the battery-backed memory bank.",
    ),
    ("question", "Who leads the VoltaArm firmware team?"),
]

# Demo cadence: automatically extract learnings every N turns (option #1). Kept small and
# in-demo on purpose — real cadence/triggering would live in the session lifecycle, not here.
AUTO_DISTILL_EVERY = 3

LESSON_QUESTION = "What must a technician do after flashing VoltaArm firmware, and why?"


# --------------------------------------------------------------------------- #
# Narration helpers                                                           #
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def step(label: str, body: str = "") -> None:
    print(f"\n  -- {label} --", flush=True)
    for line in body.splitlines():
        print(f"     {line}", flush=True)


def truncate(text: str, limit: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def answer_text(recall_result) -> str:
    """recall() returns a list of response entries; pull the answer text out."""
    if isinstance(recall_result, list) and recall_result:
        first = recall_result[0]
        return getattr(first, "answer", None) or getattr(first, "content", None) or str(first)
    return str(recall_result)


async def show_gated_guidance(user) -> None:
    """Print the active-guidance entries — these are the inputs to distillation."""
    sm = get_session_manager()
    rows = await sm.get_session_context_entries(user_id=str(user.id), session_id=SESSION_ID)
    guidance = [r for r in rows if r.get("kind", "context") == "context"]
    if not guidance:
        step("absorbed guidance (distillation inputs)", "<none absorbed yet>")
        return
    lines = [
        f"[{r.get('section')}] (conf={r.get('confidence')}) {truncate(r.get('content') or '', 90)}"
        for r in guidance
    ]
    step(
        f"absorbed guidance — {len(guidance)} gated entries (distillation inputs)", "\n".join(lines)
    )


# --------------------------------------------------------------------------- #
# (A) Seed permanent memory with remember()                                   #
# --------------------------------------------------------------------------- #
async def seed_permanent_memory() -> None:
    banner("(A) remember(documents)  ->  PERMANENT memory  (runs add -> cognify -> improve)")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    step(f"remember() {len(DOCUMENTS)} documents into dataset '{DATASET_NAME}'")
    result = await cognee.remember(DOCUMENTS, dataset_name=DATASET_NAME)
    step(
        "RememberResult (permanent mode)",
        f"status        = {result.status}\n"
        f"dataset_name  = {result.dataset_name}\n"
        f"items         = {result.items_processed}\n"
        f"-> entities + relationships are now in the graph/vector stores",
    )


# --------------------------------------------------------------------------- #
# (B+C) Multi-turn session — recall() absorbs guidance; auto-distill every N   #
# --------------------------------------------------------------------------- #
async def auto_distill_checkpoint(user, turn_no: int) -> None:
    """Automatically extract accumulated learnings into the graph (option #1).

    Fires improve(session_ids=[...]) — which runs distillation (stage 2c) over the whole
    session so far. In production this cadence would live in the session lifecycle; here it
    is a simple every-N-turns trigger for demonstration.
    """
    step(
        f"auto-extraction checkpoint (after turn {turn_no}, every {AUTO_DISTILL_EVERY} turns)",
        "calling improve(session_ids=[...]) -> distills accumulated learnings into the graph",
    )
    await cognee.improve(dataset=DATASET_NAME, session_ids=[SESSION_ID], user=user)
    await show_gated_guidance(user)


async def run_multi_turn_session(user) -> None:
    banner("(B+C) multi-turn session  ->  recall() absorbs guidance; auto-distill every N turns")
    for turn_no, (label, message) in enumerate(SESSION_TURNS, start=1):
        step(f"turn {turn_no} [{label}]", message)
        await cognee.recall(
            query_text=message,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[DATASET_NAME],
            session_id=SESSION_ID,
            user=user,
        )
        if turn_no % AUTO_DISTILL_EVERY == 0:
            await auto_distill_checkpoint(user, turn_no)


# --------------------------------------------------------------------------- #
# (D) Fresh session — the distilled lesson is now permanent                   #
# --------------------------------------------------------------------------- #
async def recall_fresh_session(user) -> None:
    banner("(D) recall(question, FRESH session)  ->  answered from PERMANENT memory alone")
    step(
        "question asked in a brand-new session (no conversation history to lean on)",
        LESSON_QUESTION,
    )
    result = await cognee.recall(
        query_text=LESSON_QUESTION,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
        session_id="verification_session",
        user=user,
    )
    step(
        "answer (long-term graph only)",
        truncate(answer_text(result))
        + "\n\n-> if this reflects the firmware/calibration lesson, distillation (triggered by"
        "\n   remember) promoted the session guidance into permanent memory.",
    )


# --------------------------------------------------------------------------- #
# (E) Visualize — distilled session-learning nodes get a gold ring            #
# --------------------------------------------------------------------------- #
async def visualize(user) -> None:
    banner("(E) visualize_graph  ->  distilled session-learning nodes marked with a gold ring")
    html = await cognee.visualize_graph(
        destination_file_path=str(VIZ_PATH),
        user=user,
    )
    has_ring = "#FFC53D" in html
    step(
        "graph rendered to HTML",
        f"file       = {VIZ_PATH}\n"
        f"memory ring= {'present (session_learnings nodes ringed in gold)' if has_ring else 'no distilled nodes found'}\n"
        "-> open the file and look for the dashed gold rings + the 'Memory' legend entry.\n"
        "   Switch the color-by mode to 'node set' to see session_learnings in gold.",
    )


async def main() -> None:
    banner(f"Logging this run to: {LOG_PATH}")
    await seed_permanent_memory()

    user = await get_default_user()
    await get_session_manager().delete_session(user_id=str(user.id), session_id=SESSION_ID)

    if not get_session_manager().is_session_available_for_completion(str(user.id)):
        print(
            "\n[!] Session cache is not available/enabled — session steps (B–C) will degrade.\n"
            "    Enable a cache backend (CACHE_BACKEND=sqlite is the default) to see them.",
            file=sys.stderr,
        )

    await run_multi_turn_session(user)
    await recall_fresh_session(user)
    await visualize(user)
    banner(f"DONE — full log saved to {LOG_PATH}\n      graph visualization: {VIZ_PATH}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        _LOG_FILE.flush()
        _LOG_FILE.close()
