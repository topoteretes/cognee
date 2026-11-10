"""Hermes MemoryProvider implementation backed by Cognee."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import (
    capture_policy,
    code_graph,
    dataset_overrides,
    exit_watcher,
    memory_credit,
    recall_notice,
    recall_text,
    recap,
    runtime_env,
    write_buffer,
)
from .backend import (
    CogneeNotInstalledError,
    MemoryBackend,
    build_backend,
    cognee_not_installed_message,
    default_backend,
    has_cognee,
)
from .config import (
    DEFAULT_DATASET,
    DEFAULT_IDENTITY_EMAIL,
    DEFAULT_IDENTITY_PASSWORD,
    DEFAULT_LOCAL_PORT,
    DEFAULT_SERVER_BOOT_TIMEOUT,
    LOCAL_COGNEE_REQUIREMENT,
    RECALL_MODE_AUTO,
    RECALL_MODE_PINNED,
    RECALL_MODE_SOURCE_ENV,
    RECALL_MODE_SOURCE_FILE,
    RECALL_MODES,
    SHARED_PLUGIN_STATE_DIR,
    config_path,
    configured_dataset_name,
    configured_recall_mode,
    effective_recall_mode,
    load_config,
    recall_mode_is_set,
    resolve_hermes_home,
    resolve_local_roots,
    save_env_vars,
    str_to_bool,
)
from .config import (
    save_config as save_plugin_config,
)
from .schemas import (
    CODE_SEARCH_SCHEMA,
    FORGET_SCHEMA,
    RECALL_SCHEMA,
    RECAP_SCHEMA,
    REMEMBER_SCHEMA,
    SWITCH_DATASET_SCHEMA,
)
from .server_bootstrap import (
    ServerStillStartingError,
    default_server_log_path,
    ensure_local_server,
    wait_for_server,
)

try:
    from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover - lets package smoke tests run outside Hermes.

    class MemoryProvider:  # type: ignore[no-redef]
        @property
        def name(self) -> str:
            raise NotImplementedError


try:
    from agent.memory_provider import spawn_context_thread
except ImportError:  # pragma: no cover - Hermes without the helper, or outside Hermes.

    def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
        return threading.Thread(target=target, args=args, kwargs=kwargs, name=name, daemon=daemon)


logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
# Cognee first, Hermes' built-in memory as the fallback. Only ever shown while
# Cognee is usable: when it is not, the provider adds nothing and the agent has
# only Hermes' memory, which is the fallback working as intended.
_MEMORY_STEER = (
    "Cognee is your primary long-term memory. For anything about the user, their "
    "people, projects or earlier work, answer from the recalled Cognee memory (the "
    "'## Cognee Memory' block on each prompt) or search it with cognee_recall. Hermes' "
    "built-in memory notes (MEMORY.md, USER.md) are the fallback, for when Cognee has "
    "nothing on the question: if they disagree with Cognee, go with Cognee and point out "
    "the difference. Store durable knowledge with cognee_remember, not the built-in "
    "memory tool."
)
_BLOCK_STEER = (
    "Recalled from Cognee, your primary memory: answer from this before Hermes' built-in "
    "memory notes, and if they disagree, go with Cognee and say so."
)
# How long session start blocks on a local server that is not answering yet. A
# fresh store boots in well under a minute; a first start on an existing store
# can migrate for many minutes, and holding the user's first message that long
# shows them only a spinner. Past this, the session starts and the provider
# finishes connecting in the background (up to the full boot timeout).
_SESSION_START_WAIT_SECS = 90.0
_BREAKER_COOLDOWN_SECS = 120

# Why recall is pinned, the hook-family plugins' ``reason`` values (claude-code
# _plugin_common.RECALL_PIN_*). Hermes has no explicit read-dataset scope, so
# ``read_dataset_ids`` never applies here.
RECALL_PIN_PICKED = "dataset_picked"  # this conversation pinned recall (pin or switch)
# The configured recall_mode is "pinned", or none is set and the configured
# dataset is not the default (config.effective_recall_mode).
RECALL_PIN_CONFIGURED = "configured_pin"


def _configured_by(source: str) -> str:
    """Where a configured pin came from, as the result message names it."""
    if source == RECALL_MODE_SOURCE_FILE:
        return 'recall_mode "pinned" in cognee.json'
    return "COGNEE_PLUGIN_RECALL_MODE=pinned"


def _recall_message(mode: str, reason: str, dataset: str, source: str = "") -> str:
    """The one human-readable line of every ``cognee_switch_dataset`` result
    (current, pin, auto, switch, reset), built from the mode and pin reason.
    ``source`` is load_config's ``recall_mode_source``, for a configured pin."""
    saves = f"Saves go to dataset {dataset!r}. "
    if mode != RECALL_MODE_PINNED:
        return saves + (
            "Recall is auto: the server picks the best-matching dataset (cognee 1.6.6; "
            "an older server searches every readable dataset). Use action='pin' to "
            "keep recall on this dataset."
        )
    if reason == RECALL_PIN_CONFIGURED and source not in (
        RECALL_MODE_SOURCE_ENV,
        RECALL_MODE_SOURCE_FILE,
    ):
        why = (
            "because it is the configured dataset and no recall mode is set "
            "(`recall_mode: auto` in cognee.json or COGNEE_PLUGIN_RECALL_MODE=auto lets "
            "Cognee pick)"
        )
        lift = "Use action='auto' to let the server pick for this conversation."
    elif reason == RECALL_PIN_CONFIGURED:
        why = f"by the configured recall mode ({_configured_by(source)})"
        lift = "Use action='auto' to let the server pick for this conversation."
    else:
        why = "by this conversation's own choice"
        lift = "Use action='auto' to let the server pick, or 'reset' to follow the config."
    return saves + f"Recall is pinned to {dataset!r} {why}. {lift}"


# cognee rejects a dataset name containing a space or a dot (check_dataset_name,
# run on every write). Only those are rewritten, so every name the server accepts
# today comes back unchanged. Shared rule: integrations/conformance/dataset_name_cases.json.
_DATASET_NAME_REJECTED_RE = re.compile(r"[ .]+")


def _safe_dataset_name(name: str, fallback: str = DEFAULT_DATASET) -> str:
    stripped = str(name or "").strip()
    cleaned = _DATASET_NAME_REJECTED_RE.sub("_", stripped)
    if not cleaned or (cleaned != stripped and not cleaned.strip("_")):
        return fallback
    return cleaned


def _safe_session_component(value: str) -> str:
    # Sanitization kept consistent with the other integrations' session-id helpers
    # (claude-code/codex `_sanitize_session_key`, openclaw `sanitizeSessionKey`):
    # keep the ASCII set [A-Za-z0-9-_.], replace others with `_`, trim `._` ends,
    # cap length at 120. `isascii()` guards against unicode letters/digits that
    # `isalnum()` would otherwise keep. The trailing `or "session"` fallback is
    # intentional and hermes-specific, so empty-output cases are excluded from the
    # shared conformance table.
    safe = "".join(
        ch if (ch.isascii() and ch.isalnum()) or ch in ("-", "_", ".") else "_" for ch in value
    )
    return safe.strip("._")[:120] or "session"


def _format_turn(user_content: str, assistant_content: str) -> str:
    return f"User: {user_content}\nAssistant: {assistant_content}"


#: Capture limits, as in openclaw (applied after redaction).
_MAX_PARAM_CHARS = 4_000
_MAX_RETURN_CHARS = 8_000
_MAX_QA_CHARS = 8_000
#: Replay budgets for buffered writes: after a write, and at session end.
_DRAIN_BUDGET_SECS = 20.0
_FINAL_DRAIN_BUDGET_SECS = 30.0


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int) -> str:
    text = _text_of(value)
    return f"{text[:limit]}…[truncated]" if len(text) > limit else text


def _tool_args(raw: Any) -> dict[str, Any]:
    """A tool call's arguments as a dict (OpenAI calls carry a JSON string)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


def _tool_error(result: Any) -> str:
    """The error a tool result reports (Hermes tools answer ``{"error": ...}``), or ""."""
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except ValueError:
            return ""
    if isinstance(data, dict) and data.get("error"):
        return _text_of(data["error"])
    return ""


def _coerce_result_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"text": value}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    # Not a dict, a string or a model: no readable fields. A ``text`` built from
    # ``str(value)`` would be an object repr (with its memory address) shown to
    # the agent as a memory (#457).
    return {}


#: Tag of the per-prompt memory block: one graph-scope ``only_context`` recall
#: rendered verbatim (see ``CogneeMemoryProvider._run_layered_prefetch``).
_MEMORY_LANE = "cognee_memory"
#: Tag of the session's distilled guidance (``scope=["session_context"]``).
_GUIDANCE_LANE = "cognee_guidance"

#: After this many prompts in a row whose recall only timed out, each prompt
#: first spends a short /health probe (openclaw's slow-server probe).
_SLOW_STREAK_THRESHOLD = 3
_READY_PROBE_TIMEOUT_SECS = 3.0


_BUDGET_EXHAUSTED_MESSAGE = (
    "Cognee %s improve skipped: the server's LLM budget is exhausted (402); not retrying"
)


def _llm_budget_exhausted(exc: Exception) -> bool:
    """cognee >= 1.6.3 answers an improve with 402 when the server's LLM budget is
    spent. The server answered, so it is no breaker failure, and nothing retries."""
    return getattr(exc, "status", None) == 402


def _server_unresponsive(exc: Exception) -> bool:
    """A write that got no answer: it timed out, or the connection broke after it
    was opened (``CogneeUnreachable`` from the HTTP transport)."""
    return _is_timeout(exc) or hasattr(exc, "not_connected")


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _readable(value: Any) -> str:
    """A text slot's value as memory text, or "" when it holds none.

    Only strings are memory: ``None``, ``0``, ``False``, ``{}`` and the like are
    not. A list of strings is joined, so its content is kept without the list's
    repr.
    """
    if isinstance(value, str):
        return value if value.strip() else ""
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        joined = "\n".join(v for v in value if v.strip())
        return joined
    return ""


def _result_text(value: Any) -> str:
    """The first readable text slot of a recall item, or "" when it has none.

    Keys are tried in order and empty or non-string values are skipped rather
    than ending the search, so ``{"text": "", "answer": "real"}`` still yields
    "real". An item with no readable text yields "" and is dropped by callers.
    It used to fall back to ``str(value)``, which injected the whole response
    envelope (or an error payload) as a memory and counted it as a hit (#457).
    """
    if isinstance(value, str):
        return _readable(value)
    data = _coerce_result_dict(value)
    for key in ("answer", "text", "content", "chunk_text", "summary"):
        found = _readable(data.get(key))
        if found:
            return found
    return ""


def _recall_failure_advice(exc: Exception) -> str:
    """One actionable sentence appended to a timeout-shaped recall failure.

    The default GRAPH_COMPLETION search runs an LLM per query, so on a local
    model a timeout is usually the search strategy, not an outage — and the
    model reading this error can fix it on the retry. Covers the HTTP
    transport (CogneeUnreachable wraps urllib's "timed out") and the SDK
    bridge (concurrent.futures.TimeoutError, empty message on 3.10).
    """
    if not isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        text = str(exc).lower()
        if "timed out" not in text and "timeout" not in text:
            return ""
    return (
        " The default GRAPH_COMPLETION search runs an LLM per query and can be "
        "slow on local models — retry with search_type='CHUNKS' (fast raw-text "
        "retrieval), or raise COGNEE_RECALL_TIMEOUT."
    )


class CogneeMemoryProvider(MemoryProvider):
    """Cognee V2/V1.1 knowledge graph memory for Hermes Agent."""

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self._config: dict[str, Any] = {}
        # How we reach cognee. An explicitly injected transport always wins;
        # otherwise initialize() picks one from config (see backend.build_backend).
        self._injected_backend = backend
        self._backend: MemoryBackend = backend or default_backend()
        self._initialized = False
        # Set while a local server that was still starting is waited for in the
        # background; _start_error holds why that wait gave up.
        self._starting = False
        self._start_error = ""
        self._remote_mode = False
        # Set once a detached worker has taken this session's improve-then-
        # unregister; from then on this process must not touch either.
        self._close_handed_off = False
        self._session_id = ""
        self._session_cognee_id = ""
        self._dataset = DEFAULT_DATASET
        self._platform = ""  # Hermes' platform for this session; shapes the credit line
        self._top_k = 5
        self._auto_route = True
        self._improve_on_end = True
        self._writes_enabled = True
        self._hermes_home: str | None = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        # Bumped on a reset session switch to invalidate prefetches still in
        # flight — see queue_prefetch / on_session_switch.
        self._prefetch_generation = 0
        self._sync_thread: Optional[threading.Thread] = None
        # The best-effort improve on_pre_compress starts; one at a time per provider.
        self._pre_compress_thread: Optional[threading.Thread] = None
        # The breaker is touched from the main thread and every worker thread
        # (prefetch, sync, memory-write); the lock keeps the failure counter's
        # read-modify-writes from losing updates under that concurrency.
        self._breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        # Set when a crash-safe exit watcher is armed (server modes only).
        self._watcher_state_path: Optional[Path] = None
        # Dataset switching: the configured default, the per-conversation switch
        # counter (feeds hermes_<id>__N cognee session ids), and sessions retired
        # by a forced switch whose bridge failed — re-submitted at session end so
        # the escape hatch defers the sync instead of dropping turns.
        self._default_dataset = DEFAULT_DATASET
        self._switch_counter = 0
        self._retired_sessions: list[tuple[str, str]] = []
        # Why graph recall is restricted to self._dataset, "" when it is not
        # (see _recall_pinned): RECALL_PIN_PICKED for this conversation's own
        # pin (action='pin' or a switch), RECALL_PIN_CONFIGURED when the
        # configured recall_mode is "pinned". Unpinned, recall is auto
        # (SDK-1393): it names no dataset and a cognee 1.6.6 server routes it to
        # the best-matching dataset(s); an older one searches every readable
        # dataset.
        self._recall_pin_reason = ""
        # This conversation's own recall mode ("auto" / "pinned"), "" when it
        # follows the configured recall_mode. The conversation's choice wins.
        self._recall_choice = ""
        # Set by initialize: the configured recall_mode is "pinned", and where
        # that setting came from (load_config's recall_mode_source).
        self._configured_pins_recall = False
        self._recall_mode_source = ""
        # Where Hermes was launched; gates the identifier-based code recall lane.
        self._cwd = ""
        # Per-session memory-hit totals for the visibility header (guarded by
        # _prefetch_lock: written by prefetch workers, read on consumption).
        self._turns_seen = 0
        self._turns_with_hits = 0
        self._hits_total = 0
        # Tags for crediting memory in the answer (memory_credit): one numbering
        # per conversation, shared by auto-recall and the cognee_recall tool.
        self._credit = memory_credit.MemoryTags()
        # Prompts in a row whose memory recall only timed out (guarded by
        # _breaker_lock); see _SLOW_STREAK_THRESHOLD.
        self._slow_streak = 0
        # Capture writes the server never received wait here (write_buffer).
        self._write_buffer_path = write_buffer.WRITE_BUFFER_PATH
        # The last capture-settings error logged, so it is logged once.
        self._capture_error_logged = ""
        # Entries stored for the current cognee session since its last
        # successful improve. A pre-compress or session-end improve with nothing
        # new is skipped (claude-code's "nothing new" rule): the server's persist
        # watermark already makes improve incremental, but each call still costs.
        self._stored_lock = threading.Lock()
        self._stored_since_improve = 0

    @property
    def name(self) -> str:
        return "cognee"

    def is_available(self) -> bool:
        # Must stay network-free: Hermes calls this for every provider during
        # discovery, before anything is started.
        cfg = load_config()
        # Remote mode is plain HTTP; only local and embedded mode run cognee here.
        if cfg.get("service_url"):
            return True
        if str_to_bool(cfg.get("embedded"), False):
            # Embedded mode runs cognee in this process, from Hermes's own
            # environment; the runtime venv is never used, so it never counts.
            return has_cognee() and not runtime_env.hermes_env_outdated()
        # No LLM-key check: a keyless provider (Ollama) needs none, and a key
        # written by `hermes memory setup` after a long-running gateway started
        # is not in that process yet — "unavailable" would hide both, while the
        # server bootstrap warns by name when it starts without one.
        return runtime_env.local_ready()

    def unavailable_reason(self) -> str:
        # Hermes skips initialize() when is_available() is False and appends this
        # to its "reports unavailable" warning, so it is the only place a user
        # whose runtime is missing (declined at setup, or dropped on upgrade from
        # 1.3.x, when cognee left Hermes's environment) learns the fix.
        cfg = load_config()
        if cfg.get("service_url"):
            return ""
        if str_to_bool(cfg.get("embedded"), False):
            if not has_cognee():
                return cognee_not_installed_message("embedded")
            if runtime_env.hermes_env_outdated():
                return runtime_env.embedded_outdated_message()
            return ""
        if runtime_env.local_ready():
            return ""
        return runtime_env.local_not_ready_message()

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "service_url",
                "description": "Cognee service URL (blank for local-server mode)",
                "required": False,
                "env_var": "COGNEE_BASE_URL",
            },
            {
                "key": "api_key",
                "description": "Cognee service API key",
                "secret": True,
                "required": False,
                "env_var": "COGNEE_API_KEY",
            },
            {
                "key": "llm_api_key",
                "description": "LLM API key for local mode (optional: blank uses local models)",
                "secret": True,
                "required": False,
                "env_var": "LLM_API_KEY",
            },
            {
                "key": "llm_model",
                "description": "LLM model for local embedded Cognee",
                "required": False,
                "env_var": "LLM_MODEL",
            },
            {
                "key": "dataset",
                "description": "Default Cognee dataset (shared with the other cognee plugins)",
                "default": DEFAULT_DATASET,
                "env_var": "COGNEE_PLUGIN_DATASET",
            },
            {
                "key": "recall_mode",
                "description": (
                    "Which datasets recall reads: auto lets Cognee pick them for each "
                    "question; pinned reads only the active dataset"
                ),
                "default": RECALL_MODE_AUTO,
                "choices": list(RECALL_MODES),
                "env_var": "COGNEE_PLUGIN_RECALL_MODE",
            },
            {
                "key": "auto_route",
                "description": (
                    "Let Cognee pick the search type for each query; false always uses "
                    "graph completion"
                ),
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "improve_on_end",
                "description": "Run Cognee improve() when a Hermes session ends",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        non_secret = {
            key: value for key, value in values.items() if key not in {"api_key", "llm_api_key"}
        }
        if non_secret:
            save_plugin_config(non_secret, hermes_home)

    def post_setup(self, hermes_home: str, config: dict[str, Any]) -> None:
        """Hermes memory setup hook for a focused Cognee setup flow."""
        print("\nCognee memory setup")
        print("-" * 40)
        print("  Deployment:")
        print("    remote - Cognee Cloud or another cognee server (default)")
        print("    local  - a cognee server on this machine")
        local_ready = runtime_env.local_ready()
        if not local_ready:
            outdated = runtime_env.outdated_requirement()
            stale = runtime_env.hermes_env_outdated()
            if outdated:
                print(f"  Local mode is off: its runtime holds {outdated}, and this version")
                print(f"  needs {LOCAL_COGNEE_REQUIREMENT}. Picking local offers to install it")
                print("  (about 1 GB from PyPI). Remote mode needs nothing installed.")
            elif stale:
                print(f"  Local mode is off: Hermes's environment holds {stale}, and this")
                print(f"  version needs {LOCAL_COGNEE_REQUIREMENT}. Picking local offers to")
                print("  install it (about 1 GB from PyPI). Remote mode needs nothing installed.")
            else:
                print("  Local mode needs a cognee runtime, which is not installed yet: picking")
                print(f"  local offers to install it ({LOCAL_COGNEE_REQUIREMENT}, about 1 GB from")
                print("  PyPI). Remote mode needs nothing installed.")

        current = load_config(hermes_home)
        if _chose_local(hermes_home):
            print("  Current mode: local (type `local` to keep it)")
        elif current.get("service_url"):
            print("  Current mode: remote")
        mode = _prompt("Mode", default="remote").strip().lower()
        remote = mode in {"remote", "cloud", "service"}

        env_values: dict[str, str] = {}
        dataset = _prompt("Dataset", default=str(current.get("dataset") or DEFAULT_DATASET))
        file_values: dict[str, Any] = {
            "dataset": dataset,
            # With no recall mode set yet, a non-default dataset offers
            # "pinned": recall stays on it, as it did before (SDK-1393).
            "recall_mode": _prompt_recall_mode(
                current.get("recall_mode")
                if recall_mode_is_set(current)
                else effective_recall_mode({}, dataset)
            ),
            # auto_route (which search type recall uses) is not asked: next to
            # the recall mode (which datasets it reads) it read as the same
            # question twice. It keeps its saved value; set it in cognee.json
            # or COGNEE_AUTO_ROUTE.
            "improve_on_end": _prompt_bool(
                "Improve graph on session end",
                current.get("improve_on_end", True),
            ),
        }

        if remote:
            service_url = _prompt(
                "Cognee service URL",
                default=str(current.get("service_url") or ""),
            )
            if service_url:
                file_values["service_url"] = service_url
                env_values["COGNEE_BASE_URL"] = service_url
                env_values["COGNEE_SERVICE_URL"] = ""  # clear deprecated alias
            api_key = _prompt_secret("Cognee API key", keep=bool(current.get("api_key")))
            if api_key:
                env_values["COGNEE_API_KEY"] = api_key
        else:
            file_values["service_url"] = ""
            # Local mode drops the remote connection whole: a Cognee Cloud key left
            # behind would be sent to the local server, which rejects it.
            env_values["COGNEE_BASE_URL"] = ""
            env_values["COGNEE_SERVICE_URL"] = ""  # clear deprecated alias
            env_values["COGNEE_API_KEY"] = ""
            llm_key = _prompt_secret(
                "LLM API key (blank: local models, no key)", keep=bool(current.get("llm_api_key"))
            )
            if llm_key:
                env_values["LLM_API_KEY"] = llm_key
            llm_model = _prompt("LLM model", default=str(current.get("llm_model") or ""))
            if llm_model:
                file_values["llm_model"] = llm_model
                env_values["LLM_MODEL"] = llm_model
            if not local_ready:
                _offer_runtime_install()

        save_plugin_config(file_values, hermes_home)
        save_env_vars(env_values, hermes_home)

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = self.name

        try:
            from hermes_cli.config import save_config

            save_config(config)
        except Exception:
            pass

        print("\n  Memory provider: cognee")
        print("  Activation saved to config.yaml")
        print("  Provider config saved to cognee.json")
        if env_values:
            print("  Secrets saved to .env")
        print("\n  Start a new chat (or `hermes` session) to use Cognee memory: a chat")
        print("  that is already open keeps the memory it started with.")
        print("  Check with: hermes memory status, hermes cognee status\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._hermes_home = kwargs.get("hermes_home")
        self._config = load_config(self._hermes_home)
        self._session_id = session_id
        configured = configured_dataset_name(self._config.get("dataset"))
        self._default_dataset = _safe_dataset_name(configured)
        if self._default_dataset != configured.strip():
            logger.warning(
                "cognee: dataset name %r is not valid for cognee; using %r",
                configured,
                self._default_dataset,
            )
        self._dataset = self._default_dataset
        # The configured recall_mode: "pinned" keeps recall on the active
        # dataset, anything else is auto. With none set, a configured dataset
        # other than the default pins (reason configured_pin), as before
        # SDK-1393; the default dataset is auto.
        self._configured_pins_recall = (
            effective_recall_mode(self._config, self._default_dataset) == RECALL_MODE_PINNED
        )
        self._recall_mode_source = str(self._config.get("recall_mode_source") or "")
        self._recall_choice = ""
        self._recall_pin_reason = self._configured_recall_pin()
        self._top_k = int(self._config.get("top_k") or 5)
        self._auto_route = str_to_bool(self._config.get("auto_route"), True)
        self._improve_on_end = str_to_bool(self._config.get("improve_on_end"), True)
        self._writes_enabled = kwargs.get("agent_context", "primary") in {"", "primary", None}
        self._platform = str(kwargs.get("platform") or "")
        self._session_cognee_id = self._build_cognee_session_id(session_id, **kwargs)
        self._apply_dataset_override()
        try:
            self._cwd = os.getcwd()
        except OSError:
            self._cwd = ""

        # Now that config is loaded, choose the transport (unless one was injected).
        if self._injected_backend is None:
            self._backend = build_backend(self._config)

        self._backend.configure_models(
            llm_api_key=str(self._config.get("llm_api_key") or ""),
            llm_model=str(self._config.get("llm_model") or ""),
        )

        # Connection mode (see README "Modes"):
        #   remote        — service_url set: thin client to a managed/cloud cognee.
        #   local-server  — default: ensure a local cognee server (single DB owner)
        #                   and connect as a thin client. No in-process DB ops, so
        #                   no "database is locked" under Hermes' background threads.
        #   embedded      — opt-in (COGNEE_EMBEDDED=true): run cognee in-process.
        #                   Single-process / offline only; the local single-writer
        #                   DBs are NOT safe under concurrent / multi-process use.
        service_url = str(self._config.get("service_url") or "")
        embedded = str_to_bool(self._config.get("embedded"), False)

        # No silent fallbacks between modes: a failure in an explicitly chosen mode
        # is surfaced. Falling back to embedded would reintroduce the exact DB-lock
        # risk this PR removes; falling back from remote to local would mask config
        # errors and silently diverge data. Embedded is reachable only on purpose
        # (COGNEE_EMBEDDED=true).
        if service_url:
            try:
                api_key = str(self._config.get("api_key") or "")
                self._backend.connect(url=service_url, api_key=api_key, timeout=30)
                self._remote_mode = True
            except Exception as exc:
                raise RuntimeError(
                    f"COGNEE_BASE_URL is set to {service_url!r} but the connection failed. "
                    "Fix the URL/network/credentials, or unset it to use local mode."
                ) from exc
        elif embedded:
            self._configure_local_roots()
            self._remote_mode = False
        else:
            try:
                # Pin the roots the other cognee plugins pin (~/.cognee by default),
                # so the store is the same no matter which plugin spawned the server
                # on this port first.
                data_root, system_root = self._local_roots()
                boot_timeout = float(
                    self._config.get("server_boot_timeout", DEFAULT_SERVER_BOOT_TIMEOUT)
                )
                local_url = ensure_local_server(
                    int(self._config.get("local_port") or DEFAULT_LOCAL_PORT),
                    data_root=data_root,
                    system_root=system_root,
                    boot_timeout=boot_timeout,
                    wait=min(boot_timeout, _SESSION_START_WAIT_SECS),
                )
                self._backend.connect(url=local_url, api_key="", timeout=30)
                self._remote_mode = True
            except CogneeNotInstalledError:
                # Already says what to do.
                raise
            except ServerStillStartingError as starting:
                logger.warning("%s Cognee memory turns on once it is ready.", starting)
                self._remote_mode = True
                self._starting = True
                spawn_context_thread(
                    self._finish_starting, args=(starting,), name="cognee-hermes-server-start"
                ).start()
                return
            except Exception as exc:
                raise RuntimeError(f"cognee local server failed to start: {exc}") from exc

        # Identity only matters in embedded mode (a local relational DB exists to
        # hold the user). In server/remote mode the server owns identity via the
        # api-key principal, and touching the local DB here would be meaningless.
        # The backend owns whatever principal it resolves; the provider never
        # handles it.
        if not self._remote_mode:
            try:
                self._backend.resolve_identity(
                    email=str(self._config.get("identity_email") or DEFAULT_IDENTITY_EMAIL),
                    password=str(
                        self._config.get("identity_password") or DEFAULT_IDENTITY_PASSWORD
                    ),
                    timeout=30,
                )
            except Exception as exc:
                logger.warning(
                    "Cognee identity initialization failed; using backend default user: %s", exc
                )

        self._complete_initialization()

    def _complete_initialization(self) -> None:
        """The connected half of initialize(); the background start runs it too."""
        self._ensure_dataset()
        self._arm_exit_watcher()
        self._initialized = True
        problem = capture_policy.capture_pattern_error(
            capture_policy.policy_from_config(self._config)
        )
        if problem:
            self._warn_capture_error(ValueError(problem))
        if self._can_buffer() and self._writes_enabled:
            # Writes an earlier run could not deliver; replayed off the startup path.
            self._replay_in_background()
        self._queue_recall_notice()

    def _queue_recall_notice(self) -> None:
        """Hold the once-per-dataset note for this session's first reply: recall
        stays on the configured dataset only because no recall mode is set, and
        auto exists (recall_notice). Not for a conversation with its own mode, or
        a subagent."""
        if (
            not self._writes_enabled
            or self._recall_choice
            or recall_mode_is_set(self._config)
            or self._recall_pin_reason != RECALL_PIN_CONFIGURED
            or not recall_notice.due(self._default_dataset)
        ):
            return
        recall_notice.queue(self._session_id, self._default_dataset)

    def _finish_starting(self, starting: ServerStillStartingError) -> None:
        """Wait out a local server that was still starting, then connect."""
        try:
            if not wait_for_server(starting):
                self._start_error = (
                    f"cognee local server at {starting.url} did not become ready. "
                    f"See {starting.log_path}."
                )
                logger.warning("%s", self._start_error)
                return
            self._backend.connect(url=starting.url, api_key="", timeout=30)
            self._complete_initialization()
            logger.info("cognee local server is ready; Cognee memory is on.")
        except Exception as exc:
            self._start_error = f"cognee local server started but connecting failed: {exc}"
            logger.warning("%s", self._start_error)
        finally:
            self._starting = False

    def _ensure_dataset(self) -> None:
        """Create-or-return the dataset up front, like the other plugins do.

        A fresh server or cloud tenant has no datasets, so a session that opens
        with a recall would otherwise hit a missing dataset. Best-effort: a write
        creates the dataset implicitly anyway, so failing here only costs the
        recall-first case — not the session.
        """
        try:
            self._backend.ensure_dataset(dataset=self._dataset, timeout=30)
        except Exception as exc:
            logger.warning("Cognee dataset ensure failed (continuing): %s", exc)

    def _arm_exit_watcher(self) -> None:
        """Crash insurance: a detached process that closes the session if we can't.

        Only meaningful when a server outlives this process (the transport says so
        via ``connection_info``); embedded mode has nothing to unregister from and
        nobody left to run an improve. Best-effort: a session must not fail to
        start because its insurance did.
        """
        try:
            info = self._backend.connection_info()
        except Exception:
            info = None
        if not info or not info.get("url"):
            return
        state_dir = SHARED_PLUGIN_STATE_DIR / "hermes"
        state_path = state_dir / "exit-watchers" / f"{os.getpid()}.json"
        try:
            exit_watcher.arm(
                state_path=state_path,
                log_path=state_dir / "exit-watcher.log",
                parent_pid=os.getpid(),
                url=str(info.get("url") or ""),
                api_key=str(info.get("api_key") or ""),
                agent_session_name=str(info.get("agent_session_name") or "hermes"),
                dataset=self._dataset,
                session_id=self._session_cognee_id,
                improve=bool(self._writes_enabled and self._improve_on_end),
                improve_timeout=self._timeout("improve_timeout", 300),
            )
            self._watcher_state_path = state_path
        except Exception as exc:
            logger.debug("could not arm the cognee exit watcher: %s", exc)

    def _is_usable(self) -> bool:
        """False when initialization never completed, so nothing may touch cognee.

        ``MemoryManager.initialize_all`` catches and logs whatever ``initialize()``
        raises, then carries on — Hermes starts either way and the tool schemas
        stay registered. Without this gate the provider would keep serving calls
        against an unconnected transport: for the SDK that means falling back to
        in-process cognee, which is exactly the single-writer DB risk local-server
        mode exists to avoid. Fail closed and say so instead.
        """
        return self._initialized

    def _unavailable_envelope(self) -> str:
        if self._starting:
            message = (
                "Cognee memory is not ready yet: the local cognee server is still starting "
                "(a first start on an existing store migrates it, which can take several "
                "minutes). It turns on in this session once the server is ready; see "
                f"{default_server_log_path()} for progress."
            )
        elif self._start_error:
            message = f"Cognee memory is unavailable: {self._start_error}"
        else:
            message = (
                "Cognee memory is unavailable: initialization did not complete. "
                "Check the Hermes logs for the cause and `hermes cognee status` "
                "for the current configuration."
            )
        return json.dumps({"error": message})

    def system_prompt_block(self) -> str:
        # Never advertise memory the model cannot actually use.
        if not self._is_usable():
            return ""
        mode = "remote" if self._remote_mode else "local"
        lines = [
            "# Cognee Memory",
            f"Active ({mode}). Dataset: {self._dataset}.",
            "Use cognee_recall for prior context, cognee_remember for durable facts, "
            "cognee_forget when the user asks to remove Cognee memory, "
            "cognee_switch_dataset to move this conversation to another dataset, "
            "cognee_code_search for structural questions about an indexed repository, "
            "and cognee_recap for a standup, digest or timeline of recorded sessions.",
        ]
        # The memory steer — the counterpart of claude-code's COGNEE_PREFER_MEMORY
        # and openclaw's memorySteer: without it the agent reaches for the host's
        # native memory files by habit and the graph never hears about it.
        if str_to_bool(self._config.get("memory_steer"), True):
            steer = str(self._config.get("memory_steer_text") or "").strip() or _MEMORY_STEER
            lines.append(steer)
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._is_usable():
            return ""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        # The system prompt's steer is far from the answer, and Hermes' own memory
        # notes sit right beside it; the same rule next to the recalled memory is
        # what keeps an agent from answering from a stale built-in note.
        if str_to_bool(self._config.get("memory_steer"), True):
            return f"## Cognee Memory\n{_BLOCK_STEER}\n{result}"
        return f"## Cognee Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._is_usable() or not query or self._is_breaker_open():
            return

        cognee_session_id = self._session_cognee_id_for(session_id)
        # Snapshot the generation this prefetch belongs to. A reset session
        # switch bumps it, so a recall that was already in flight for the
        # previous conversation discards its result instead of landing in the
        # fresh one. Joining the worker on switch is not enough: a recall slower
        # than any bounded join would still write after the clear.
        generation = self._prefetch_generation

        self._prefetch_thread = threading.Thread(
            target=self._run_layered_prefetch,
            args=(query, cognee_session_id, generation),
            daemon=True,
            name="cognee-hermes-prefetch",
        )
        self._prefetch_thread.start()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Capture a completed turn: its tool calls as traces, then the turn.

        ``messages`` (Hermes passes the conversation as of the turn) carries the
        assistant's tool calls and their results; each becomes a ``trace``
        entry, under the capture policy (``capture_tools`` / ``capture_deny_tools``
        / secret-file paths) and with secrets redacted, as the claude-code, codex
        and openclaw plugins capture them. Redaction runs before truncation. A
        capture setting that cannot be parsed fails closed: nothing is stored.

        A write the server never received is queued and replayed later
        (``write_buffer``); while the breaker is open, writes go straight to
        the queue.

        The answer's memory-credit line is checked against the tags this
        session handed out, then stripped: the credit says where the answer
        came from and must not be stored back as part of it.
        """
        if memory_credit.has_credit(assistant_content or ""):
            result = self._credit.check(assistant_content)
            logger.info("cognee memory_credit %s", json.dumps(result, ensure_ascii=False))
        assistant_content = memory_credit.strip_credit_lines(
            recall_notice.strip_notice(assistant_content or "")
        )
        if not self._is_usable() or not self._writes_enabled:
            return
        settings = capture_policy.policy_from_config(self._config)
        if not capture_policy.capture_enabled(settings):
            return
        cognee_session_id = self._session_cognee_id_for(session_id)
        dataset = self._dataset
        try:
            question = _clip(capture_policy.redact(user_content or "", settings), _MAX_QA_CHARS)
            answer = _clip(capture_policy.redact(assistant_content or "", settings), _MAX_QA_CHARS)
            traces = self._turn_traces(messages, settings)
        except Exception as exc:  # a capture pattern that does not parse: fail closed
            self._warn_capture_error(exc)
            return
        writes: list[tuple[str, dict[str, Any]]] = [("trace", entry) for entry in traces]
        writes.append(
            (
                "qa",
                # The prompt and the answer in their own fields, as the other
                # plugins store them: recaps list the prompts, and the timeline
                # finds a topic in them.
                {"type": "qa", "question": question, "answer": answer, "context": ""},
            )
        )

        if self._is_breaker_open():
            if self._can_buffer():
                for kind, entry in writes:
                    self._buffer_write(dataset, cognee_session_id, kind, entry)
            return

        def _sync() -> None:
            self._write_turn(writes, cognee_session_id, dataset)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync,
            daemon=True,
            name="cognee-hermes-sync",
        )
        self._sync_thread.start()

    def _write_turn(
        self, writes: list[tuple[str, dict[str, Any]]], cognee_session_id: str, dataset: str
    ) -> None:
        """Send one turn's traces and its QA entry under one deadline, one verdict.

        The whole turn gets ``write_timeout``; each write gets what is left of
        it, so a server that accepts connections but never answers holds this
        thread for one write timeout, not one per tool call. The first sign that
        the server is away (a bufferable failure, a timeout or reset, the
        breaker opening, the deadline passing) ends the turn: what was not sent
        yet is queued (``write_buffer``), since it certainly never reached the
        server. A write that timed out is not queued, since the server may have
        stored it. A write the server answered with an error is dropped and the
        rest still go, because the server is answering.

        The breaker hears one verdict per turn, as for recall: one failure when
        the server was away, too slow for the deadline, or refused any write; a
        success (and a replay of the queue) only when the whole turn was stored.
        A turn the breaker cut short gives no verdict of its own. A turn whose
        traces landed but whose QA entry failed is a failure, never a success.
        """
        deadline = time.monotonic() + self._timeout("write_timeout", 120)
        can_buffer = self._can_buffer()
        pending = list(writes)
        stored = False
        refused = False
        outage: Optional[Exception] = None
        out_of_time = False
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out_of_time = True
                break
            if self._is_breaker_open():
                break
            kind, entry = pending[0]
            try:
                self._send_entry(kind, entry, cognee_session_id, dataset, timeout=remaining)
                pending.pop(0)
                stored = True
                self._note_stored(cognee_session_id)
            except Exception as exc:
                if kind == "trace" and not can_buffer:
                    # The in-process SDK has no typed-entry write; the turn
                    # itself is still stored.
                    pending.pop(0)
                    logger.debug("Cognee trace capture unavailable: %s", exc)
                    continue
                if can_buffer and write_buffer.is_bufferable_write_error(exc):
                    outage = exc  # this entry never reached the server: queued below
                    break
                pending.pop(0)
                if _server_unresponsive(exc):
                    outage = exc
                    logger.warning(
                        "Cognee %s write got no answer; not retried, the server may "
                        "have stored it: %s",
                        kind,
                        exc,
                    )
                    break
                refused = True
                logger.warning("Cognee session sync failed (%s): %s", kind, exc)

        if pending:
            if can_buffer:
                for kind, entry in pending:
                    self._buffer_write(dataset, cognee_session_id, kind, entry)
                logger.info(
                    "Cognee buffered %d write(s) until the server is reachable%s",
                    len(pending),
                    f": {outage}" if outage else "",
                )
            else:
                logger.warning(
                    "Cognee dropped %d write(s) of this turn: the server did not answer",
                    len(pending),
                )

        if outage is not None or refused or out_of_time:
            self._record_failure()
        elif stored and not pending:
            self._record_success()
            # The server answers again: replay anything an outage left behind,
            # on its own thread, so this turn's thread (which sync_turn, forget
            # and the switches join) is free again at once.
            self._replay_in_background()

    def _send_entry(
        self,
        kind: str,
        entry: dict[str, Any],
        cognee_session_id: str,
        dataset: str,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        write_timeout = self._timeout("write_timeout", 120)
        timeout = write_timeout if timeout is None else max(0.1, min(write_timeout, timeout))
        if kind == "qa" and not self._can_buffer():
            # The in-process SDK has no typed entries: the turn goes in as one
            # framed text through remember_session.
            self._backend.remember_session(
                text=_format_turn(str(entry.get("question") or ""), str(entry.get("answer") or "")),
                session_id=cognee_session_id,
                dataset=dataset,
                timeout=timeout,
            )
            return
        self._backend.remember_entry(
            entry=entry, session_id=cognee_session_id, dataset=dataset, timeout=timeout
        )

    def _turn_traces(
        self, messages: Optional[list[dict[str, Any]]], settings: dict[str, str]
    ) -> list[dict[str, Any]]:
        """``trace`` entries for the tool calls of the turn that just completed.

        The turn starts after the last user message. Raises
        ``CapturePatternError`` (or ``re.error``) when a capture setting cannot
        be parsed, so the caller stores nothing.
        """
        if not messages:
            return []
        start = 0
        for index, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                start = index + 1
        turn = [m for m in messages[start:] if isinstance(m, dict)]
        results = {
            str(m.get("tool_call_id") or ""): m.get("content")
            for m in turn
            if m.get("role") == "tool"
        }
        traces: list[dict[str, Any]] = []
        for message in turn:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = str(function.get("name") or call.get("name") or "")
                if not name:
                    continue
                params = _tool_args(function.get("arguments"))
                if capture_policy.is_self_reference(name, params):
                    continue
                if not capture_policy.allow_tool(name, params, settings):
                    continue
                result = results.get(str(call.get("id") or ""))
                error = _tool_error(result)
                redacted = capture_policy.redact(params, settings)
                traces.append(
                    {
                        "type": "trace",
                        "origin_function": name,
                        "status": "error" if error else "success",
                        "method_params": {
                            str(key): _clip(value, _MAX_PARAM_CHARS)
                            for key, value in redacted.items()
                        },
                        "method_return_value": _clip(
                            capture_policy.redact(_text_of(result), settings), _MAX_RETURN_CHARS
                        ),
                        "error_message": _clip(
                            capture_policy.redact(error, settings), _MAX_PARAM_CHARS
                        ),
                        # LLM feedback per step is expensive on a busy session;
                        # the server-side AUTO_FEEDBACK + improve pass covers it.
                        "generate_feedback_with_llm": False,
                    }
                )
        return traces

    def _warn_capture_error(self, exc: Exception) -> None:
        message = f"Cognee capture is off until the capture settings are fixed: {exc}"
        if message != self._capture_error_logged:
            self._capture_error_logged = message
            logger.warning(message)

    def _can_buffer(self) -> bool:
        """Whether writes reach a server that outlives this process (HTTP)."""
        try:
            return bool(self._backend.connection_info())
        except Exception:
            return False

    def _buffer_write(self, dataset: str, session_id: str, kind: str, entry: dict) -> None:
        try:
            write_buffer.write_buffer_for(self._write_buffer_path).enqueue(
                dataset=dataset, session_id=session_id, kind=kind, entry=entry
            )
        except Exception as exc:
            logger.warning("Cognee %s could not be buffered: %s", kind, exc)

    def _drain_write_buffer(self, budget_secs: float, session_id: str = "") -> int:
        """Replay buffered writes; ``session_id`` limits it to one session."""
        if not self._can_buffer():
            return 0
        buffer = write_buffer.write_buffer_for(self._write_buffer_path)
        if not session_id and not buffer.maybe_non_empty:
            return 0

        def send(item: dict[str, Any], remaining: float) -> None:
            item_session = str(item.get("session_id") or "")
            # What is left of the drain's budget, never more than one write's.
            self._backend.remember_entry(
                entry=item.get("entry") or {},
                session_id=item_session,
                dataset=str(item.get("dataset") or self._dataset),
                timeout=max(0.1, min(self._timeout("write_timeout", 120), remaining)),
            )
            self._note_stored(item_session)

        def dropped(item: dict[str, Any], exc: BaseException) -> None:
            logger.warning(
                "Cognee dropped a buffered %s for %s: %s",
                item.get("kind"),
                item.get("session_id"),
                exc,
            )

        try:
            stored = buffer.drain(
                send, budget_secs=budget_secs, session_id=session_id, on_drop=dropped
            )
        except Exception as exc:
            logger.warning("Cognee buffered write replay failed: %s", exc)
            return 0
        if stored:
            logger.info("Cognee replayed %d buffered capture write(s)", stored)
        return stored

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = [RECALL_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]
        # Config may not be loaded yet when Hermes collects schemas; the
        # defaults (on) match load_config's, so both paths agree.
        if str_to_bool(self._config.get("dataset_switch_tool"), True):
            schemas.append(SWITCH_DATASET_SCHEMA)
        if str_to_bool(self._config.get("code_search_tool"), True):
            schemas.append(CODE_SEARCH_SCHEMA)
        if str_to_bool(self._config.get("recap_tool"), True):
            schemas.append(RECAP_SCHEMA)
        return schemas

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._is_usable():
            return self._unavailable_envelope()
        if self._is_breaker_open():
            return json.dumps(
                {
                    "error": (
                        "Cognee is temporarily unavailable after repeated failures. "
                        "The provider will retry automatically after cooldown."
                    )
                }
            )
        if tool_name == "cognee_recall":
            return self._handle_recall(args)
        if tool_name == "cognee_remember":
            return self._handle_remember(args)
        if tool_name == "cognee_forget":
            return self._handle_forget(args)
        if tool_name == "cognee_switch_dataset":
            return self._handle_switch_dataset(args)
        if tool_name == "cognee_code_search":
            return self._handle_code_search(args)
        if tool_name == "cognee_recap":
            return self._handle_recap(args)
        return json.dumps({"error": f"Unknown Cognee tool: {tool_name}"})

    def _replay_in_background(self) -> None:
        """Replay the write buffer on a daemon thread; ``drain`` is single-flight."""
        if not self._can_buffer():
            return
        if not write_buffer.write_buffer_for(self._write_buffer_path).maybe_non_empty:
            return
        threading.Thread(
            target=self._drain_write_buffer,
            args=(_DRAIN_BUDGET_SECS,),
            daemon=True,
            name="cognee-hermes-replay",
        ).start()

    def _note_stored(self, cognee_session_id: str, count: int = 1) -> None:
        """Count an entry that reached the current session's cache."""
        if cognee_session_id and cognee_session_id == self._session_cognee_id:
            with self._stored_lock:
                self._stored_since_improve += count

    def _stored_snapshot(self) -> int:
        with self._stored_lock:
            return self._stored_since_improve

    def _mark_improved(self, snapshot: int) -> None:
        """An improve bridged ``snapshot`` entries; later ones still count."""
        with self._stored_lock:
            self._stored_since_improve = max(0, self._stored_since_improve - snapshot)

    def _reset_stored(self) -> None:
        """A different cognee session: nothing of it has been stored yet."""
        with self._stored_lock:
            self._stored_since_improve = 0

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        if (
            not self._is_usable()
            or not self._writes_enabled
            or not self._improve_on_end
            or self._is_breaker_open()
        ):
            return
        # Turns buffered while the server was away must be in the session cache
        # before improve bridges it, or they never reach the graph.
        self._drain_write_buffer(_FINAL_DRAIN_BUDGET_SECS, session_id=self._session_cognee_id)
        # Sessions retired by a forced dataset switch get their deferred bridge
        # before the current session's own close.
        self._bridge_retired_sessions()
        # Nothing stored since the last improve (a compaction just bridged it,
        # or the session wrote nothing): the close still unregisters, but no
        # improve is paid for.
        nothing_new = self._stored_snapshot() == 0
        if nothing_new:
            logger.debug("Cognee session-end improve skipped: nothing new since the last one")
        if self._hand_off_session_close(improve=not nothing_new):
            return
        if not nothing_new:
            self._improve_inline()

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Bridge the session into the graph before Hermes compresses it away.

        Best-effort (Hermes pre-compress checkpoint API v1): the improve runs on a
        worker so compression never waits on a graph build, a failure is logged and
        never raised, and no text is contributed to the summary prompt. The
        session-end improve still runs; this only stops long sessions from losing
        compressed-away turns that never reached the graph.
        """
        try:
            if (
                not messages
                or not self._session_cognee_id
                or not self._is_usable()
                or not self._writes_enabled
                or not self._improve_on_end
                or self._is_breaker_open()
            ):
                return ""
            if self._pre_compress_thread and self._pre_compress_thread.is_alive():
                return ""  # the previous compaction's improve is still running
            dataset, session_id, sync_thread = (
                self._dataset,
                self._session_cognee_id,
                self._sync_thread,
            )

            def _improve() -> None:
                # The latest turn has to be in the session cache before it is bridged.
                if sync_thread and sync_thread.is_alive():
                    sync_thread.join(timeout=10.0)
                # So must the turns buffered while the server was away.
                self._drain_write_buffer(_DRAIN_BUDGET_SECS, session_id=session_id)
                snapshot = self._stored_snapshot()
                if snapshot == 0:
                    # Nothing stored since the last improve: nothing to bridge.
                    logger.debug("Cognee pre-compress improve skipped: nothing new")
                    return
                try:
                    self._backend.improve(
                        dataset=dataset,
                        session_ids=[session_id],
                        background=False,
                        timeout=self._timeout("improve_timeout", 300),
                    )
                    self._record_success()
                    self._mark_improved(snapshot)
                except Exception as exc:
                    if _llm_budget_exhausted(exc):
                        logger.warning(_BUDGET_EXHAUSTED_MESSAGE, "pre-compress")
                        return
                    self._record_failure()
                    logger.warning("Cognee pre-compress improve failed: %s", exc)

            self._pre_compress_thread = threading.Thread(
                target=_improve, daemon=True, name="cognee-hermes-pre-compress"
            )
            self._pre_compress_thread.start()
        except Exception as exc:
            logger.warning("Cognee pre-compress improve failed: %s", exc)
        return ""

    def _hand_off_session_close(self, *, improve: bool = True) -> bool:
        """Give improve-then-unregister to a detached worker. True when it took them.

        Two requirements that cannot both hold in this process: the user's exit
        must not wait on a graph build, and the improve must finish *before* the
        agent unregisters — the local server runs with COGNEE_AGENT_MODE=true and
        its watchdog SIGTERMs the server within 60s of the agent count reaching
        zero, running pipelines or not. Outside this process both hold at once:
        the worker blocks on the improve, and its blocking costs nobody anything.
        This is how the claude-code, codex and openclaw plugins close a session.
        """
        if self._watcher_state_path is None:
            return False  # embedded mode, or arming failed — nobody to hand to
        if str(self._config.get("improve_background") or "").strip():
            return False  # an explicit override asked for the in-process path
        try:
            info = self._backend.connection_info() or {}
            handed = exit_watcher.finalize(
                self._watcher_state_path,
                api_key=str(info.get("api_key") or ""),
                session_id=self._session_cognee_id,
                dataset=self._dataset,
                improve=improve,
                improve_timeout=self._timeout("improve_timeout", 300),
            )
        except Exception as exc:
            logger.debug("could not hand off the cognee session close: %s", exc)
            return False
        self._close_handed_off = handed
        return handed

    def _improve_inline(self) -> None:
        """Bridge the session here, because no detached worker will.

        Synchronous by default: this is either embedded mode, where the work runs
        in this process and dies with it, or a failed handoff, where shutdown() is
        about to unregister and take the server with it.
        ``COGNEE_IMPROVE_BACKGROUND=true`` opts out for a server nothing here can
        shut down.
        """
        background = str_to_bool(self._config.get("improve_background"), False)
        snapshot = self._stored_snapshot()
        try:
            self._backend.improve(
                dataset=self._dataset,
                session_ids=[self._session_cognee_id],
                background=background,
                timeout=self._timeout("improve_timeout", 300),
            )
            self._record_success()
            self._mark_improved(snapshot)
            # The bridge ran; a crash after this point should not improve again.
            # The watcher stays armed for the unregister half.
            if self._watcher_state_path is not None:
                exit_watcher.update(self._watcher_state_path, improve=False)
        except Exception as exc:
            if _llm_budget_exhausted(exc):
                logger.warning(_BUDGET_EXHAUSTED_MESSAGE, "session-end")
                return
            self._record_failure()
            logger.warning("Cognee session-end improve failed: %s", exc)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._session_id = new_session_id
        # A different conversation means its own dataset override (or none) and
        # its own switch counter — never the previous conversation's.
        self._dataset = self._default_dataset
        self._switch_counter = 0
        self._recall_choice = ""
        self._recall_pin_reason = self._configured_recall_pin()
        self._session_cognee_id = self._build_cognee_session_id(new_session_id, **kwargs)
        self._reset_stored()
        self._apply_dataset_override()
        if self._watcher_state_path is not None and not self._close_handed_off:
            # Re-point the crash insurance at the new session (and re-enable the
            # improve half in case a previous session end disabled it). Skipped
            # after a handoff: that state file describes the session a detached
            # worker is still closing, and is its to delete.
            exit_watcher.update(
                self._watcher_state_path,
                session_id=self._session_cognee_id,
                dataset=self._dataset,
                improve=bool(self._writes_enabled and self._improve_on_end),
            )
        if reset:
            with self._prefetch_lock:
                self._prefetch_result = ""
                # Invalidate any recall still in flight for the old conversation.
                # Only on reset: /resume, /branch and compression continue the same
                # logical conversation, so a prefetch issued for it stays valid.
                self._prefetch_generation += 1
                # The hit totals describe one conversation; a fresh one starts at 0.
                self._turns_seen = 0
                self._turns_with_hits = 0
                self._hits_total = 0
                self._credit = memory_credit.MemoryTags()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        # ``_writes_enabled`` is checked here for the same reason sync_turn checks
        # it: a non-primary agent_context (a subagent) must not write to memory.
        # Without it a subagent's built-in memory write still reached the graph
        # while its conversation turns were correctly suppressed.
        if (
            not self._is_usable()
            or not self._writes_enabled
            or action not in {"add", "replace"}
            or not content
            or self._is_breaker_open()
        ):
            return
        metadata = dict(metadata or {})
        source = metadata.get("write_origin") or "hermes_memory_tool"
        # A mirror of Hermes' own memory writes is automatic capture, so the
        # capture policy's redaction applies (explicit cognee_remember is not).
        settings = capture_policy.policy_from_config(self._config)
        if not capture_policy.capture_enabled(settings):
            return
        try:
            content = capture_policy.redact(content, settings)
        except Exception as exc:  # a redaction pattern that does not parse: fail closed
            self._warn_capture_error(exc)
            return
        payload = f"Hermes {target} memory ({action}, {source}): {content}"

        def _sync() -> None:
            try:
                self._remember_permanent(payload, self._dataset)
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.debug("Cognee memory-write mirror failed: %s", exc)

        threading.Thread(target=_sync, daemon=True, name="cognee-hermes-memory-write").start()

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        if not task and not result:
            return
        content = f"Delegated task: {task}\nResult: {result}"
        self.sync_turn(content, "", session_id=self._session_id)

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread, self._pre_compress_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        if self._close_handed_off:
            # A detached worker owns improve-then-unregister for this session.
            # Unregistering here would drop the agent count to zero while that
            # improve is still running, and COGNEE_AGENT_MODE's watchdog would
            # retire the server mid-promotion — the exact race the handoff
            # removes. The worker owns the state file too, so nothing to disarm.
            self._backend.close(timeout=5, unregister=False)
            return
        if self._watcher_state_path is not None:
            # No handoff, so this is the orderly path — stand the insurance down
            # BEFORE closing the backend, so the watcher can never race the
            # unregister that is about to happen.
            exit_watcher.disarm(self._watcher_state_path)
            self._watcher_state_path = None
        self._backend.close(timeout=5)

    def _build_cognee_session_id(self, session_id: str, **kwargs) -> str:
        # Convention across integrations: "{agent}_{native_session_id}" —
        # e.g. hermes_<hermes-session-id>. (kwargs like agent_workspace/user_id are
        # accepted for call-site compatibility but no longer embedded in the name.)
        prefix = str(self._config.get("session_prefix") or "hermes")
        return f"{prefix}_{_safe_session_component(session_id)}"

    def _session_cognee_id_for(self, session_id: str) -> str:
        if not session_id or session_id == self._session_id:
            return self._session_cognee_id
        return self._build_cognee_session_id(session_id)

    def _local_roots(self) -> tuple[str, str]:
        return resolve_local_roots(self._config)

    def _configure_local_roots(self) -> None:
        """Point the backend at local storage (in-process transports only)."""
        data_root, system_root = self._local_roots()
        self._backend.configure_local_roots(data_root=data_root, system_root=system_root)

    def backup_paths(self) -> list[str]:
        """Cognee storage that ``hermes backup`` would otherwise miss.

        ``hermes backup`` walks ``HERMES_HOME`` on its own, so only roots living
        elsewhere need declaring. By default that is all of them: local storage
        sits in the shared ``~/.cognee`` (see ``resolve_local_roots``), so a
        backup deliberately includes the machine's shared memory store. Must work
        without ``initialize()`` and without network, so it reads config directly.
        """
        config = load_config()
        home = resolve_hermes_home()
        external: list[str] = []
        for root in resolve_local_roots(config):
            path = Path(root).expanduser()
            if home is not None and path.is_relative_to(home):
                continue
            external.append(str(path))
        return external

    def _timeout(self, key: str, default: float) -> float:
        """A named, bounded timeout read from config at call time."""
        return float(self._config.get(key, default))

    def _recall(
        self,
        query: str,
        *,
        search_type: Any,
        top_k: int,
        session_id: str,
        datasets: Optional[list[str]] = None,
    ) -> list[Any]:
        """The explicit ``cognee_recall`` search: the knowledge graph, stated outright.

        Memory is read from the graph only. The session cache (the server's
        ``session``, ``trace`` and ``session_context`` scopes) and the ``auto``
        scope that folds it in are never requested — those entries are noise
        next to the cognified graph. No session id travels, as in openclaw's
        ``memory_search``: on cognee >= 1.6.0 it would put this conversation's
        own turns, which Hermes already holds, into the item's prompt.
        ``search_type`` is the caller's override or None for the query
        classifier. ``session_id`` is accepted for call-site compatibility.
        """
        del session_id
        return self._backend.recall(
            query=query,
            session_id=None,
            datasets=datasets if datasets is not None else self._graph_datasets(),
            top_k=top_k,
            auto_route=self._auto_route,
            query_type=search_type or None,
            scope=["graph"],
            timeout=self._timeout("recall_timeout", 120),
        )

    def _remember_permanent(self, content: str, dataset: str) -> Any:
        kwargs: dict[str, Any] = {}
        # Opt-in (#428): unset leaves chunking to the server, as before.
        chunk_size = self._char_limit("chunk_size", 0)
        if chunk_size > 0:
            kwargs["chunk_size"] = chunk_size
        return self._backend.remember_permanent(
            text=content,
            dataset=dataset,
            session_ids=[self._session_cognee_id],
            timeout=self._timeout("write_timeout", 120),
            **kwargs,
        )

    # -- layered per-prompt recall -------------------------------------------

    def _hit_header(self, hits: int) -> str:
        """One plain-words line on what memory just contributed, or "".

        Must be called under ``_prefetch_lock`` after the counters were updated:
        the per-session totals it renders are the ones this turn just advanced.
        """
        if not str_to_bool(self._config.get("memory_hits"), True):
            return ""
        line = f"{hits} memory hit{'s' if hits != 1 else ''} this turn"
        line += f" · {self._turns_with_hits}/{self._turns_seen} turns had hits this session"
        return line + "\n"

    def _code_lane(self, query: str) -> dict[str, Any]:
        """The additive code-graph lane for this prompt, or {}.

        Syntactically gated: fires only when the prompt names an
        identifier-shaped token AND either the cwd sits inside a repo indexed
        via ``hermes cognee index-repo`` or ``code_datasets`` names one.
        """
        if not str_to_bool(self._config.get("code_graph_recall"), True):
            return {}
        try:
            lane = code_graph.auto_code_lane(query, self._cwd)
        except Exception as exc:
            logger.debug("code lane gate failed: %s", exc)
            lane = {}
        if lane:
            return lane
        extra = [
            name.strip()
            for name in str(self._config.get("code_datasets") or "").split(",")
            if name.strip()
        ]
        if not extra:
            return {}
        identifiers = code_graph.extract_identifiers(query)
        if not identifiers:
            return {}
        return {
            "dataset": extra[0],
            "identifier": identifiers[0],
            "code_query": code_graph.build_code_query(identifiers[0]),
        }

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        """A 404 on the graph or guidance scope means nothing was cognified (or
        distilled) yet — routine on a fresh install, so it must not read as a
        recall failure."""
        return getattr(exc, "status", None) == 404

    def _run_layered_prefetch(self, query: str, session_id: str, generation: int) -> None:
        """Up to three lanes per prompt, injecting only what Hermes does not have.

        * Memory: one graph-scope ``HYBRID_COMPLETION`` recall with
          ``only_context``, sent WITHOUT the session id (SDK-904). On cognee
          >= 1.6.0 the item's ``text`` is the full LLM input the completion would
          have read; with a session id that input also carries this
          conversation's recent turns, which Hermes already holds and which were
          most of a mid-session payload. :func:`recall_text.trim_recall_text`
          keeps only the retrieved context: the question template goes,
          passages recorded in this session go, each passage is capped at
          ``recall_passage_chars`` and the item at ``recall_context_chars``.
          Text that does not parse (an older server's bare context) is injected
          whole.
        * Guidance: the session's distilled guidance, from its own
          ``scope=["session_context"]`` request with the session id against the
          capture dataset, injected as ``<cognee_guidance>``. A 404 means none
          exists yet.
        * Code: the deterministic ``code_query`` lane when a prompt arms it.

        The lanes are dispatched together under one deadline; a lane that fails
        or misses it is dropped without costing the others. ``recall_context_chars``
        then caps the whole injection (guidance first, then memory, then code),
        the same rules as the claude-code, codex and openclaw plugins.
        """
        # A new turn before anything can fail: a prompt whose recall returns
        # early has no tags of its own.
        credit = self._credit
        credit.begin_turn()
        budget = self._config.get("recall_budget")
        deadline = time.monotonic() + (20.0 if budget is None else float(budget))
        recall_timeout = self._timeout("recall_timeout", 120)
        top_k = min(self._top_k, 5)
        passage_chars = self._char_limit("recall_passage_chars", recall_text.DEFAULT_PASSAGE_CHARS)
        context_chars = self._char_limit("recall_context_chars", recall_text.DEFAULT_CONTEXT_CHARS)

        # Slow-server probe: after a few prompts in a row whose recall only
        # timed out, a short /health check decides whether to try at all,
        # instead of spending the whole budget on a server that is not
        # answering. Any answered recall ends the streak.
        if self._slow_streak >= _SLOW_STREAK_THRESHOLD and not self._server_answers_quickly():
            logger.info(
                "Cognee server not responding (%d slow recalls in a row); skipping recall",
                self._slow_streak,
            )
            with self._prefetch_lock:
                self._turns_seen += 1
            return

        lanes: list[tuple[str, dict[str, Any]]] = []
        code_lane = self._code_lane(query)
        if code_lane:
            lanes.append(
                (
                    "code_graph",
                    {
                        "scope": ["code"],
                        "datasets": [code_lane["dataset"]],
                        "code_query": code_lane["code_query"],
                    },
                )
            )
        routed = self._graph_datasets() is None
        # HYBRID_COMPLETION combines BM25 + vector + graph retrieval; with
        # only_context the LLM completion is skipped server-side and the item's
        # text is the prompt that completion would have read.
        lanes.append(
            (
                _MEMORY_LANE,
                {
                    "scope": ["graph"],
                    "query_type": "HYBRID_COMPLETION",
                    "session_id": None,
                    "routed": routed,
                },
            )
        )
        if session_id:
            lanes.append(
                (
                    _GUIDANCE_LANE,
                    {"scope": ["session_context"], "session_id": session_id},
                )
            )

        # Same deadline for every lane: min(per-call timeout, budget left).
        # Below the floor a call cannot return anything useful, so nothing is
        # dispatched rather than firing requests with a doomed deadline.
        remaining = deadline - time.monotonic()
        if remaining < 0.2:
            lanes = []
        lane_timeout = min(recall_timeout, max(remaining, 0.0))

        def _run_lane(spec: dict[str, Any]) -> tuple[Any, Optional[Exception]]:
            """One lane's call; never raises, so no lane can take down the rest."""
            try:
                results = self._backend.recall(
                    query=query,
                    session_id=spec.get("session_id"),
                    # A routed lane names no dataset: the server picks (auto recall).
                    datasets=(
                        None if spec.get("routed") else spec.get("datasets") or [self._dataset]
                    ),
                    top_k=top_k,
                    auto_route=True,
                    query_type=spec.get("query_type"),
                    scope=spec["scope"],
                    code_query=spec.get("code_query"),
                    only_context=True,
                    timeout=lane_timeout,
                )
            except Exception as exc:
                return None, exc
            return results, None

        outcomes: list[tuple[Any, Optional[Exception]]] = []
        if lanes:
            # One worker per lane. The HTTP backend blocks on its socket; the
            # SDK backend hands every call to its single dedicated event loop,
            # where the lanes interleave as coroutines. Each call bounds itself
            # with ``lane_timeout``, so the pool joins within the budget.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(lanes), thread_name_prefix="cognee-recall-lane"
            ) as pool:
                outcomes = list(pool.map(_run_lane, [spec for _label, spec in lanes]))

        memories: list[str] = []
        memory_datasets: list[str] = []  # the dataset each memory came from, if named
        code_blocks: list[str] = []
        guidance = ""
        item_guidance = ""
        answered = False
        hard_failures = 0
        memory_timed_out = False
        for (label, spec), (results, exc) in zip(lanes, outcomes):
            if exc is not None:
                if self._is_not_found(exc) and spec["scope"] in (["graph"], ["session_context"]):
                    # Nothing cognified yet, or no guidance yet: routine on a
                    # fresh install, not a recall failure.
                    answered = True
                    continue
                if label == _GUIDANCE_LANE:
                    # Guidance is a side lane: its failure says nothing about
                    # whether memory works, so it never feeds the breaker.
                    logger.debug("Cognee guidance recall failed: %s", exc)
                    continue
                if label == _MEMORY_LANE and _is_timeout(exc):
                    memory_timed_out = True
                hard_failures += 1
                logger.debug("Cognee recall lane %s failed: %s", label, exc)
                continue
            answered = True
            if label == _MEMORY_LANE:
                for text, dataset in self._memory_lane_items(results):
                    context, item_guide, stats = recall_text.trim_recall_text(
                        text,
                        query,
                        session_id,
                        passage_chars=passage_chars,
                        context_chars=context_chars,
                    )
                    event = "trimmed" if stats.get("parsed") else "unparsed"
                    logger.debug("cognee recall_context_%s %s", event, json.dumps(stats))
                    if item_guide and not item_guidance:
                        item_guidance = item_guide
                    if context.strip():
                        memories.append(context.strip())
                        memory_datasets.append(dataset)
            elif label == _GUIDANCE_LANE:
                guidance = "\n\n".join(self._memory_lane_texts(results)).strip()
            else:
                lines = self._format_recall_lines(results, limit=top_k)
                if lines:
                    code_blocks.append(f"<{label}>\n" + "\n".join(lines) + f"\n</{label}>")

        # One verdict per turn, not per lane: a single dead server must not
        # feed the breaker several failures every prompt, and one healthy lane
        # is proof the backend is up.
        if answered:
            self._record_success()
        elif hard_failures:
            self._record_failure()
        with self._breaker_lock:
            if memory_timed_out and not answered:
                self._slow_streak += 1
            elif answered:
                self._slow_streak = 0

        # A guidance block can trail the graph item even without a session id;
        # the guidance request renders the same lines, so it wins.
        guidance = guidance or item_guidance.strip()
        capped = recall_text.cap_total(
            [guidance, *memories, *code_blocks],
            context_chars,
            lambda index: index > len(memories),
        )
        kept_guidance = capped[0]
        kept = [
            (text, dataset)
            for text, dataset in zip(capped[1 : 1 + len(memories)], memory_datasets)
            if text.strip()
        ]
        kept_memories = [text for text, _dataset in kept]
        kept_code = [block for block in capped[1 + len(memories) :] if block.strip()]

        cite = memory_credit.credit_enabled()
        if cite:
            kept_memories = [credit.tag_graph(text, dataset) for text, dataset in kept]
            kept_code = [f"{credit.label('code', 'code graph', b)}\n{b}" for b in kept_code]
            if kept_guidance.strip():
                guide = kept_guidance.strip()
                kept_guidance = f"{credit.label('guidance', 'agent guidance', guide)}\n{guide}"

        blocks = [f"<{_MEMORY_LANE}>\n{text}\n</{_MEMORY_LANE}>" for text in kept_memories]
        blocks.extend(kept_code)
        if kept_guidance.strip():
            blocks.append(f"<{_GUIDANCE_LANE}>\n{kept_guidance.strip()}\n</{_GUIDANCE_LANE}>")
        hits = len(kept_memories) + sum(block.count("\n- ") for block in kept_code)

        with self._prefetch_lock:
            self._turns_seen += 1
            if blocks:
                if hits:
                    self._turns_with_hits += 1
                    self._hits_total += hits
                if generation == self._prefetch_generation:
                    header = self._hit_header(hits) if hits else ""
                    instruction = ""
                    if cite and credit.found:
                        instruction = (
                            memory_credit.credit_instruction(
                                not credit.credited,
                                len(credit.found),
                                # Routed recall may answer from any dataset: the
                                # line names the ones the used items' tags name.
                                "" if routed else self._dataset,
                                self._platform,
                                datasets=credit.credit_datasets() if routed else (),
                            )
                            + "\n\n"
                        )
                    self._prefetch_result = instruction + header + "\n\n".join(blocks)

    def _char_limit(self, key: str, default: int) -> int:
        """A non-negative character limit from config; 0 disables it."""
        try:
            value = int(self._config.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    def _server_answers_quickly(self) -> bool:
        try:
            return bool(self._backend.probe_health(timeout=_READY_PROBE_TIMEOUT_SECS))
        except Exception:
            return False

    def _handle_recall(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "Missing required parameter: query"})
        top_k = min(max(1, int(args.get("top_k") or self._top_k)), 20)
        # A ``scope`` argument from an older tool schema is ignored: every
        # explicit recall targets the graph (see ``_recall``).
        search_type = args.get("search_type")

        try:
            # None: auto recall, routed server-side, so the credit names no dataset.
            datasets = self._graph_datasets()
            results = self._recall(
                query,
                search_type=search_type,
                top_k=top_k,
                session_id=self._session_cognee_id,
                datasets=datasets,
            )
            self._record_success()
            # Items with no readable text are not memories: dropping them keeps
            # an all-empty response a clean miss instead of empty "hits" (#457).
            items = [
                normalized
                for normalized in (self._normalize_recall_item(item) for item in results)
                if normalized["text"].strip()
            ]
            overflow = self._backend.overflow_hint()
            if not items:
                # Distinguish a genuine miss from a backend condition that makes
                # recall structurally unable to match (e.g. an embedder change
                # leaving stored and query vectors different sizes). A confirmed
                # cause is a hard error rather than a silent empty result.
                dim_message = self._backend.empty_recall_hint()
                if dim_message:
                    return json.dumps({"error": dim_message, "count": 0})
                if overflow:
                    return json.dumps({"error": overflow, "count": 0})
                return json.dumps({"result": "No relevant Cognee memory found.", "count": 0})
            envelope: dict[str, Any] = {"results": items, "count": len(items)}
            if memory_credit.credit_enabled():
                named = []
                for item in items:
                    origin = (
                        memory_credit.passage_origin(item["text"])
                        if recall_text.passage_session_id(item["text"])
                        else "Cognee recall"
                    )
                    dataset = str(item.get("dataset_name") or "").strip()
                    named.append(dataset)
                    origin = memory_credit.with_dataset(origin, dataset)
                    item["tag"] = self._credit.label("recall", origin, item["text"])
                # Auto recall names the datasets the results came from, when the
                # server named one for every result.
                routed_datasets = (
                    list(dict.fromkeys(named)) if datasets is None and all(named) else []
                )
                envelope["credit"] = memory_credit.tool_credit_instruction(
                    self._dataset if datasets else "", self._platform, routed_datasets
                )
            if overflow:
                # Non-empty results still warn: mean-pooled vectors match
                # *something*, so plausible-but-wrong hits are the overflow's
                # most misleading symptom.
                envelope["warning"] = overflow
            return json.dumps(envelope)
        except Exception as exc:
            self._record_failure()
            return json.dumps(
                {"error": f"Cognee recall failed: {exc}{_recall_failure_advice(exc)}"}
            )

    def _handle_remember(self, args: dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return json.dumps({"error": "Missing required parameter: content"})
        dataset = str(args.get("dataset") or self._dataset)

        try:
            result = self._remember_permanent(content, dataset)
            self._record_success()
            status = getattr(result, "status", "completed")
            envelope = {"result": "Content stored in Cognee.", "status": str(status)}
            # Embedding runs server-side after this write returns, so a hint here
            # usually reports the *previous* write's overflow — still worth
            # surfacing, since the index is already degrading either way.
            overflow = self._backend.overflow_hint()
            if overflow:
                envelope["warning"] = overflow
            return json.dumps(envelope)
        except Exception as exc:
            self._record_failure()
            return json.dumps({"error": f"Cognee remember failed: {exc}"})

    _FORGET_MAX_SCANNED = 100
    _FORGET_MAX_CANDIDATES = 8
    _FORGET_PREVIEW_CHARS = 240

    def _resolve_dataset_id(self, name: str, *, timeout: float = 30.0) -> str:
        """The UUID of the named dataset, or "" when this principal cannot see it."""
        for row in self._backend.list_datasets(timeout=timeout):
            if str(row.get("name") or "") == name:
                return str(row.get("id") or "")
        return ""

    @staticmethod
    def _forget_terms(raw_terms: str) -> list[str]:
        return [token for token in raw_terms.lower().replace(",", " ").split() if len(token) >= 2]

    def _handle_forget(self, args: dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip().lower()
        if action == "find":
            return self._forget_find(args)
        if action == "forget":
            return self._forget_delete(args)
        return json.dumps(
            {
                "error": (
                    "Specify action='find' (list candidate documents by terms) or "
                    "action='forget' (delete confirmed data_ids)."
                )
            }
        )

    def _forget_find(self, args: dict[str, Any]) -> str:
        terms = self._forget_terms(str(args.get("terms") or ""))
        if not terms:
            return json.dumps({"error": "action='find' requires terms describing the content."})
        dataset = str(args.get("dataset") or self._dataset)
        # Flush the turn currently being written so it is at least in the cache;
        # note that unbridged session turns are not documents yet either way.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        try:
            dataset_id = self._resolve_dataset_id(dataset)
            if not dataset_id:
                return json.dumps({"error": f"Dataset {dataset!r} was not found."})
            items = self._backend.list_dataset_data(dataset_id=dataset_id, timeout=30.0)
            candidates: list[dict[str, Any]] = []
            scanned = 0
            # The doc cap alone is not a time bound — 100 raw reads at the
            # per-read timeout could hold the tool call for minutes. One overall
            # deadline; a partial scan is reported as such via `scanned`.
            scan_deadline = time.monotonic() + self._timeout("write_timeout", 120)
            for item in items[: self._FORGET_MAX_SCANNED]:
                data_id = str(item.get("id") or "")
                if not data_id:
                    continue
                if time.monotonic() >= scan_deadline:
                    break
                scanned += 1
                try:
                    raw = self._backend.read_raw_data(
                        dataset_id=dataset_id,
                        data_id=data_id,
                        timeout=min(30.0, max(1.0, scan_deadline - time.monotonic())),
                    )
                except Exception as exc:
                    logger.debug("raw read of %s failed during forget find: %s", data_id, exc)
                    continue
                lowered = raw.lower()
                matched = [term for term in terms if term in lowered]
                if not matched:
                    continue
                first = lowered.find(matched[0])
                start = max(0, first - self._FORGET_PREVIEW_CHARS // 3)
                preview = raw[start : start + self._FORGET_PREVIEW_CHARS].strip()
                candidates.append(
                    {
                        "data_id": data_id,
                        "name": str(item.get("name") or ""),
                        "matched_terms": matched,
                        "preview": preview,
                    }
                )
            self._record_success()
            candidates.sort(key=lambda c: len(c["matched_terms"]), reverse=True)
            envelope: dict[str, Any] = {
                "action": "find",
                "dataset": dataset,
                "dataset_id": dataset_id,
                "candidates": candidates[: self._FORGET_MAX_CANDIDATES],
                "scanned": scanned,
                "total_items": len(items),
                "note": (
                    "Show these candidates to the user; delete only what they confirm, "
                    "via action='forget' with data_ids and confirm=true. Turns from the "
                    "current conversation become deletable documents only after the "
                    "session is bridged to the graph."
                ),
            }
            if not candidates:
                envelope["result"] = "No stored documents matched those terms."
            return json.dumps(envelope)
        except Exception as exc:
            self._record_failure()
            return json.dumps({"error": f"Cognee forget find failed: {exc}"})

    def _forget_delete(self, args: dict[str, Any]) -> str:
        if not bool(args.get("confirm", False)):
            return json.dumps(
                {"error": "Deletion requires confirm=true, after the user has confirmed."}
            )
        dataset = str(args.get("dataset") or self._dataset)
        if bool(args.get("everything_in_dataset", False)):
            # Whole-dataset deletion stays on the coarse endpoint; the everything
            # (all datasets) wipe is deliberately not expressible from this tool.
            try:
                result = self._backend.forget(
                    dataset=dataset,
                    everything=False,
                    memory_only=False,
                    timeout=self._timeout("write_timeout", 120),
                )
                self._record_success()
                return json.dumps({"result": f"Dataset {dataset!r} deleted.", "details": result})
            except Exception as exc:
                self._record_failure()
                return json.dumps({"error": f"Cognee forget failed: {exc}"})

        data_ids = [str(value) for value in (args.get("data_ids") or []) if str(value).strip()]
        if not data_ids:
            return json.dumps({"error": "action='forget' requires data_ids (from action='find')."})
        try:
            dataset_id = self._resolve_dataset_id(dataset)
            if not dataset_id:
                return json.dumps({"error": f"Dataset {dataset!r} was not found."})
        except Exception as exc:
            self._record_failure()
            return json.dumps({"error": f"Cognee forget failed: {exc}"})

        deleted: list[str] = []
        errors: list[dict[str, str]] = []
        for data_id in data_ids:
            try:
                self._backend.forget_document(
                    dataset_id=dataset_id,
                    data_id=data_id,
                    timeout=self._timeout("write_timeout", 120),
                )
                deleted.append(data_id)
            except Exception as exc:
                if getattr(exc, "status", None) == 404:
                    errors.append({"data_id": data_id, "error": "not found (already deleted?)"})
                else:
                    errors.append({"data_id": data_id, "error": str(exc)[:200]})
        if deleted:
            self._record_success()
        elif errors:
            self._record_failure()
        envelope: dict[str, Any] = {
            "action": "forget",
            "dataset": dataset,
            "deleted": deleted,
            "count": len(deleted),
        }
        if errors:
            envelope["errors"] = errors
        return json.dumps(envelope)

    # -- dataset switching -----------------------------------------------------

    def _apply_dataset_override(self) -> None:
        """Re-apply this conversation's persisted recall mode and dataset switch.

        The mode comes from ``recall``; the dataset only from a real switch
        entry (a dataset and a counter above 0). A mode stored alone applies to
        the active dataset, which follows the configuration.
        """
        override = dataset_overrides.load_override(self._session_id)
        if not override:
            return
        dataset = str(override.get("dataset") or "")
        try:
            counter = int(override.get("counter") or 0)
        except (TypeError, ValueError):
            counter = 0
        switched = bool(dataset) and counter > 0
        recall = override.get("recall")
        if recall in RECALL_MODES:
            self._recall_choice = str(recall)
        elif recall is None and switched and dataset != self._default_dataset:
            # Entries written before SDK-1393 carry no mode: a switch pinned recall.
            self._recall_choice = RECALL_MODE_PINNED
        else:
            # A reset (RECALL_FOLLOWS_CONFIG), or a pre-SDK-1393 entry on the
            # configured dataset, which an old file cannot tell from a reset:
            # recall follows the configuration.
            self._recall_choice = ""
        self._recall_pin_reason = self._recall_reason_for(self._recall_choice)
        if not switched:
            return
        self._dataset = dataset
        self._switch_counter = counter
        self._session_cognee_id = f"{self._session_cognee_id}__{counter}"
        self._reset_stored()

    def _handle_switch_dataset(self, args: dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip().lower()
        force = bool(args.get("force", False))
        if action == "current":
            mode, reason = self._recall_mode_and_reason()
            return json.dumps(
                {
                    "dataset": self._dataset,
                    "default": self._default_dataset,
                    "switched": self._dataset != self._default_dataset,
                    "recall": mode,
                    **({"reason": reason} if reason else {}),
                    "cognee_session_id": self._session_cognee_id,
                    "message": _recall_message(
                        mode, reason, self._dataset, self._recall_mode_source
                    ),
                }
            )
        if action == "auto":
            return self._set_recall_auto()
        if action == "pin":
            return self._pin_recall()
        if action == "list":
            try:
                rows = self._backend.list_datasets(timeout=30.0)
                self._record_success()
            except Exception as exc:
                self._record_failure()
                return json.dumps({"error": f"Listing datasets failed: {exc}"})
            names = sorted({str(row.get("name") or "") for row in rows} - {""})
            return json.dumps(
                {
                    "current": self._dataset,
                    "default": self._default_dataset,
                    "datasets": [
                        {"name": name, "current": name == self._dataset} for name in names
                    ],
                    "note": (
                        "A name not listed is created on switch. Code-graph datasets "
                        "(codebase-*) belong to indexed repositories — do not move "
                        "conversations into them."
                    ),
                }
            )
        if action == "switch":
            target = str(args.get("dataset") or "").strip()
            if not target:
                return json.dumps({"error": "action='switch' requires a dataset name."})
            if target.lower() == "auto":
                return json.dumps(
                    {
                        "error": (
                            "'auto' is reserved for the recall mode, not a dataset name: use "
                            "action='auto' to let the server pick the dataset for recall."
                        )
                    }
                )
            # Refused rather than rewritten, so a switch never lands in a dataset
            # the user did not name.
            suggestion = _safe_dataset_name(target, fallback="")
            if suggestion != target:
                hint = f" Try {suggestion!r}." if suggestion else ""
                error = f"Invalid dataset name {target!r}: cognee rejects spaces and dots."
                return json.dumps({"error": error + hint})
            return self._switch_dataset(target, force=force)
        if action == "reset":
            if self._dataset == self._default_dataset:
                return self._reset_recall()
            result = json.loads(self._switch_dataset(self._default_dataset, force=force))
            if result.get("switched"):
                # Back to the configured dataset means back to the configured
                # recall mode.
                self._follow_configured_recall()
                mode, reason = self._recall_mode_and_reason()
                result["recall"] = mode
                result.pop("recall_reason", None)
                if reason:
                    result["recall_reason"] = reason
                result["message"] = _recall_message(
                    mode, reason, self._dataset, self._recall_mode_source
                )
            return json.dumps(result)
        return json.dumps({"error": "Specify action: list, current, switch, pin, auto, or reset."})

    # -- recall mode (SDK-1393) -------------------------------------------------

    def _graph_datasets(self) -> Optional[list[str]]:
        """The datasets graph recall names: none (auto, routed server-side)
        unless recall is pinned. A server before cognee 1.6.6 does not route:
        there an unpinned recall searches every dataset the user can read."""
        return [self._dataset] if self._recall_pinned else None

    @property
    def _recall_pinned(self) -> bool:
        """True when graph recall is restricted to self._dataset, for any reason."""
        return bool(self._recall_pin_reason)

    @_recall_pinned.setter
    def _recall_pinned(self, pinned: bool) -> None:
        # In memory only (tests): this conversation's own pin, or its own auto.
        self._recall_choice = RECALL_MODE_PINNED if pinned else RECALL_MODE_AUTO
        self._recall_pin_reason = self._recall_reason_for(self._recall_choice)

    def _configured_recall_pin(self) -> str:
        """The pin the configured recall_mode alone gives, "" for auto."""
        return RECALL_PIN_CONFIGURED if self._configured_pins_recall else ""

    def _recall_reason_for(self, choice: str) -> str:
        """The pin reason for a conversation choice ("" follows the config)."""
        if choice == RECALL_MODE_PINNED:
            return RECALL_PIN_PICKED
        if choice == RECALL_MODE_AUTO:
            return ""
        return self._configured_recall_pin()

    def _recall_mode_and_reason(self) -> tuple[str, str]:
        """The mode in effect and, when pinned, why (the hook family's reasons)."""
        if self._recall_pin_reason:
            return RECALL_MODE_PINNED, self._recall_pin_reason
        return RECALL_MODE_AUTO, ""

    def _invalidate_prefetch(self) -> None:
        with self._prefetch_lock:
            self._prefetch_result = ""
            self._prefetch_generation += 1

    def _choose_recall(self, choice: str) -> None:
        """Make ``choice`` ("auto", "pinned", or "" to follow the config) this
        conversation's recall mode and persist it.

        After a switch the entry keeps the switch (dataset and counter); without
        one it stores the mode only, so recall applies to the active dataset,
        which follows the configuration. Following the config with no switch
        leaves nothing to store.
        """
        self._recall_choice = choice
        self._recall_pin_reason = self._recall_reason_for(choice)
        if self._switch_counter > 0:
            dataset_overrides.save_override(
                self._session_id,
                self._dataset,
                self._switch_counter,
                recall=choice or dataset_overrides.RECALL_FOLLOWS_CONFIG,
            )
        elif choice:
            dataset_overrides.save_override(self._session_id, None, 0, recall=choice)
        else:
            dataset_overrides.clear_override(self._session_id)
        self._invalidate_prefetch()

    def _recall_envelope(self, reason: str, **extra: Any) -> str:
        """A no-switch reply (pin, auto, reset on the configured dataset): the
        mode in effect, why, and the one message built from them."""
        mode, why = self._recall_mode_and_reason()
        envelope: dict[str, Any] = {
            "switched": False,
            "reason": reason,
            "dataset": self._dataset,
            "recall": mode,
        }
        if why:
            envelope["recall_reason"] = why
        envelope.update(extra)
        envelope["message"] = _recall_message(mode, why, self._dataset, self._recall_mode_source)
        return json.dumps(envelope)

    def _set_recall_auto(self) -> str:
        """``auto``: save auto as this conversation's choice, so it also wins
        over a later configured ``pinned``; the reply reports the mode in effect."""
        lifted = self._recall_pin_reason
        if self._recall_choice != RECALL_MODE_AUTO:
            self._choose_recall(RECALL_MODE_AUTO)
        extra = {}
        if lifted == RECALL_PIN_CONFIGURED:
            # Named so the agent can say the lift lasts this conversation only.
            extra["lifted"] = lifted
        return self._recall_envelope("recall_auto" if lifted else "already_auto", **extra)

    def _pin_recall(self) -> str:
        """``pin`` (or a switch to the dataset in use): pin recall to the active
        dataset, no bridge and no new session. Saved as this conversation's own
        mode even when the configured recall_mode already pins: it then wins
        over a later configured ``auto``. The mode is stored without the
        dataset, so recall stays on whatever dataset is active (after a config
        change, the new configured one unless this conversation switched)."""
        already = self._recall_pinned
        if self._recall_choice != RECALL_MODE_PINNED:
            self._choose_recall(RECALL_MODE_PINNED)
        return self._recall_envelope("already_active" if already else "recall_restricted")

    def _follow_configured_recall(self) -> None:
        """Drop this conversation's own recall choice: recall follows the config."""
        self._choose_recall("")

    def _reset_recall(self) -> str:
        """``reset`` on the configured dataset: recall goes back to the
        configured recall_mode."""
        before = self._recall_pin_reason
        self._follow_configured_recall()
        changed = before != self._recall_pin_reason
        return self._recall_envelope("recall_reset" if changed else "already_default")

    def _switch_dataset(self, target: str, *, force: bool) -> str:
        if target == self._dataset:
            # The dataset in use: a switch to it is a pin.
            return self._pin_recall()
        old_dataset, old_session = self._dataset, self._session_cognee_id

        # 1. Flush the turn currently being written, then bridge the session we
        #    are leaving into its dataset. A cognee session never spans two
        #    datasets, so this is the last chance its cached turns have of
        #    reaching the old dataset's graph. run_in_background: the server owns
        #    the promotion; the tool call must not stall on a graph build.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        bridged = True
        bridge_error = ""
        if self._writes_enabled and self._improve_on_end:
            try:
                self._backend.improve(
                    dataset=old_dataset,
                    session_ids=[old_session],
                    background=True,
                    timeout=self._timeout("improve_timeout", 300),
                )
            except Exception as exc:
                bridged, bridge_error = False, str(exc)[:300]
                if not force:
                    self._record_failure()
                    return json.dumps(
                        {
                            "error": (
                                f"Bridging the current session into {old_dataset!r} failed, "
                                "so nothing was switched. Retry, or pass force=true to "
                                "switch anyway (the session is then re-bridged at session "
                                f"end). Cause: {bridge_error}"
                            )
                        }
                    )
                # Forced past the failure: defer the bridge instead of dropping it.
                self._retired_sessions.append((old_dataset, old_session))

        # 2. Make sure the target exists for this principal (idempotent).
        try:
            self._backend.ensure_dataset(dataset=target, timeout=30.0)
        except Exception as exc:
            self._record_failure()
            return json.dumps(
                {"error": f"Dataset {target!r} is not available to this principal: {exc}"}
            )

        # 3. Re-point capture, recall and the session-end improve under a fresh
        #    cognee session id.
        self._switch_counter += 1
        base = self._build_cognee_session_id(self._session_id)
        self._session_cognee_id = f"{base}__{self._switch_counter}"
        self._reset_stored()
        self._dataset = target
        # A switch pins recall to the target.
        self._choose_recall(RECALL_MODE_PINNED)
        if self._watcher_state_path is not None and not self._close_handed_off:
            exit_watcher.update(
                self._watcher_state_path,
                session_id=self._session_cognee_id,
                dataset=target,
                improve=bool(self._writes_enabled and self._improve_on_end),
            )
        self._record_success()
        mode, why = self._recall_mode_and_reason()
        envelope: dict[str, Any] = {
            "switched": True,
            "dataset": target,
            "recall": mode,
            "recall_reason": why,
            "cognee_session_id": self._session_cognee_id,
            "previous": {"dataset": old_dataset, "session_id": old_session, "bridged": bridged},
        }
        if bridge_error:
            envelope["previous"]["bridge_error"] = bridge_error
        envelope["message"] = _recall_message(mode, why, target, self._recall_mode_source)
        return json.dumps(envelope)

    def _bridge_retired_sessions(self) -> None:
        """Re-submit the bridge for sessions retired by a forced switch."""
        if not self._retired_sessions:
            return
        remaining: list[tuple[str, str]] = []
        for dataset, session_id in self._retired_sessions:
            try:
                self._backend.improve(
                    dataset=dataset,
                    session_ids=[session_id],
                    background=True,
                    timeout=self._timeout("improve_timeout", 300),
                )
            except Exception as exc:
                logger.warning(
                    "re-bridging retired session %s into %s failed: %s", session_id, dataset, exc
                )
                remaining.append((dataset, session_id))
        self._retired_sessions = remaining

    # -- code graph --------------------------------------------------------------

    def _resolve_code_dataset(self, repo: str) -> str:
        """Map a repo path/URL/dataset name — or the cwd — to a code dataset."""
        repo = repo.strip()
        configured = [
            name.strip()
            for name in str(self._config.get("code_datasets") or "").split(",")
            if name.strip()
        ]
        if repo:
            if repo.startswith("codebase-") or repo in configured:
                return repo
            state = code_graph.find_repo_state(repo)
            return str(state.get("dataset") or "")
        state = code_graph.find_indexed_repo(self._cwd)
        if state.get("dataset"):
            return str(state["dataset"])
        return configured[0] if configured else ""

    def _handle_code_search(self, args: dict[str, Any]) -> str:
        operation = str(args.get("operation") or "").strip().lower()
        if operation not in code_graph.CODE_OPERATIONS:
            return json.dumps(
                {
                    "error": f"Unknown operation {operation!r}. One of: "
                    + ", ".join(code_graph.CODE_OPERATIONS)
                }
            )
        name = str(args.get("name") or "").strip()
        target = str(args.get("target") or "").strip()
        limit = min(max(1, int(args.get("limit") or 10)), 50)

        dataset = self._resolve_code_dataset(str(args.get("repo") or ""))
        if not dataset:
            return json.dumps(
                {
                    "error": (
                        "No indexed repository found. Index one first with "
                        "`hermes cognee index-repo <path-or-url>`, or name an indexed "
                        "repo/dataset via the repo parameter."
                    )
                }
            )

        code_query: dict[str, Any] = {"operation": operation}
        if operation == "query_facts":
            if name:
                code_query["name"] = name
            code_query["limit"] = limit
        elif operation == "explore":
            if not name:
                return json.dumps({"error": "explore requires name (the symbol to explore)."})
            code_query["name"] = name
        elif operation == "traverse":
            if not name:
                return json.dumps({"error": "traverse requires name (the seed symbol)."})
            code_query["start"] = name
        elif operation == "find_path":
            if not name or not target:
                return json.dumps({"error": "find_path requires name (source) and target."})
            code_query["source"] = name
            code_query["target"] = target
        elif operation == "impact_analysis":
            if not name:
                return json.dumps({"error": "impact_analysis requires name (the symbol)."})
            code_query["targets"] = [name]

        try:
            results = self._backend.recall(
                query=name or operation,
                session_id=None,
                datasets=[dataset],
                top_k=limit,
                auto_route=True,
                query_type=None,
                scope=["code"],
                code_query=code_query,
                only_context=True,
                timeout=self._timeout("recall_timeout", 120),
            )
            self._record_success()
        except Exception as exc:
            self._record_failure()
            return json.dumps({"error": f"Cognee code search failed: {exc}"})
        items = [_coerce_result_dict(item) for item in results]
        return json.dumps({"dataset": dataset, "results": items, "count": len(items)})

    def _handle_recap(self, args: dict[str, Any]) -> str:
        """A standup, digest or timeline of this dataset's recorded sessions.

        Only sessions recorded into the active dataset are listed, so on a
        shared server a recap never shows another user's conversations;
        ``all_sessions`` adds other clients' sessions in that dataset.
        """
        kind = str(args.get("kind") or "").strip().lower()
        if kind not in recap.RECAP_KINDS:
            return json.dumps({"error": f"kind must be one of {', '.join(recap.RECAP_KINDS)}"})
        projects = args.get("projects")
        try:
            dataset_ids = recap.resolve_dataset_ids(self._backend, [self._dataset])
            result = recap.run_recap(
                recap.BackendRecapApi(self._backend),
                kind=kind,
                dataset_ids=dataset_ids,
                datasets=[self._dataset],
                topic=str(args.get("topic") or ""),
                since=str(args.get("since") or ""),
                projects=[p for p in projects if isinstance(p, str)]
                if isinstance(projects, list)
                else None,
                all_sessions=args.get("all_sessions") is True,
                session_prefixes=(f"{self._config.get('session_prefix') or 'hermes'}_",),
            )
        except Exception as exc:
            return json.dumps({"kind": kind, "error": f"Cognee recap failed: {exc}"})
        return json.dumps(
            {"kind": kind, "markdown": result["markdown"], "warnings": result["warnings"]}
        )

    def _normalize_recall_item(self, item: Any) -> dict[str, Any]:
        data = _coerce_result_dict(item)
        normalized = {
            "text": _result_text(item),
            "source": data.get("source") or data.get("_source") or "cognee",
        }
        for key in ("score", "dataset", "dataset_name", "node_name"):
            if data.get(key) is not None:
                normalized[key] = data[key]
        return normalized

    def _memory_lane_texts(self, results: list[Any]) -> list[str]:
        """Each graph item's ``text`` verbatim — that string *is* the memory.

        On cognee >= 1.6.0 an ``only_context`` completion item is the whole
        LLM input: history first, retrieved context in the middle, guidance at
        the end. Truncating or bulleting it would cut exactly what memory is
        for, so nothing is parsed, stripped or shortened. ``system_prompt`` is
        never read. Older servers put the bare context in ``text``, which
        passes through the same way.
        """
        return [text for text, _dataset in self._memory_lane_items(results)]

    def _memory_lane_items(self, results: list[Any]) -> list[tuple[str, str]]:
        """``_memory_lane_texts`` with the dataset the server says each came from
        (``dataset_name`` on the search result; blank when it names none)."""
        items = []
        for item in results:
            data = _coerce_result_dict(item)
            # The text slot is used only when it holds real text; otherwise the
            # ordered fallback runs, and an item with nothing readable is
            # dropped instead of stringified into the block (#457).
            text = _readable(data.get("text")).strip() or _result_text(item).strip()
            if text:
                items.append((text, str(data.get("dataset_name") or "").strip()))
        return items

    def _format_recall_lines(self, results: list[Any], *, limit: int) -> list[str]:
        lines = []
        for item in results[:limit]:
            normalized = self._normalize_recall_item(item)
            text = normalized.get("text", "").strip()
            if not text:
                continue
            source = normalized.get("source", "cognee")
            lines.append(f"- [{source}] {text[:500]}")
        return lines

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            tripped = self._consecutive_failures >= _BREAKER_THRESHOLD
            if tripped:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
                failures = self._consecutive_failures
        if tripped:
            logger.warning(
                "Cognee circuit breaker tripped after %d consecutive failures; pausing for %ds.",
                failures,
                _BREAKER_COOLDOWN_SECS,
            )


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        value = ""
    return value or (default or "")


def _chose_local(hermes_home: str) -> bool:
    """True when an earlier setup saved local mode (an explicit empty service_url)."""
    path = config_path(hermes_home)
    try:
        saved = json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return False
    return isinstance(saved, dict) and saved.get("service_url") == ""


def _offer_runtime_install() -> None:
    """Local branch of setup: build the runtime venv now, with the user's yes.

    When a local server this plugin started is running, the question says the
    install stops it (``install-runtime`` asks the same on its own)."""
    label = f"Install the local runtime now ({LOCAL_COGNEE_REQUIREMENT}, about 1 GB from PyPI)"
    servers = runtime_env.servers_to_stop()
    if servers:
        pids = ", ".join(str(pid) for pid, _port in servers)
        label += f"? This stops the running local server (pid {pids})."
    if not _prompt_bool(label, True):
        print("  Skipped. Install it later with: hermes cognee install-runtime")
        return
    try:
        runtime_env.install_runtime()
    except runtime_env.RuntimeInstallError as exc:
        print(f"  Could not install the local runtime: {exc}")
        print("  Retry with: hermes cognee install-runtime")


def _prompt_recall_mode(current: Any) -> str:
    """The setup wizard's recall-mode question; an unknown answer is auto.
    ``current`` is the default offered."""
    print("  Recall mode: which datasets recall reads. auto lets Cognee pick the")
    print("  datasets for each question; pinned reads only the active dataset.")
    answer = _prompt("Recall mode (auto/pinned)", default=configured_recall_mode(current))
    return configured_recall_mode(answer)


def _prompt_bool(label: str, default: bool) -> bool:
    default_text = "y" if default else "n"
    value = _prompt(f"{label} (y/n)", default=default_text).strip().lower()
    return value in {"y", "yes", "true", "1", "on"}


def _prompt_secret(label: str, *, keep: bool) -> str:
    import getpass
    import sys

    suffix = " (blank to keep current)" if keep else ""
    try:
        if sys.stdin.isatty():
            return getpass.getpass(f"  {label}{suffix}: ").strip()
        return input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
