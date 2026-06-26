"""reconcile_memory — RECONCILE / supersede: detect contradictory claims in memory, link the stale claim
to the current one with a ``supersedes`` edge, and demote the stale claim so retrieval prefers current truth.

Closes #3394 (part of the #3389 memory-transformation umbrella).

This is the "self-correcting wiki" move several hackathon teams built by hand (SUPERSEDES edges, retired
claims), made a first-class memify task. It deliberately composes existing primitives only —
``get_graph_data``, ``add_edge``, ``get_node_feedback_weights`` / ``set_node_feedback_weights`` — plus one
``LLMGateway`` structured-output judgment for the contradiction decision.

Design (matches the ticket):
- **Detect:** candidate pairs are nodes that share a neighbour (i.e. claims about the same subject) — a
  deterministic, bounded pre-filter so the LLM never judges the full O(n^2). Each candidate pair is judged
  by ``LLMGateway`` (does claim A contradict claim B + a confidence).
- **Resolve:** pick the *current* claim (config ``prefer``: ``recency`` by ``updated_at``/``created_at``, or
  ``feedback`` by ``feedback_weight``), add a ``supersedes`` edge **current → stale**, and demote the stale
  node's ``feedback_weight``. The demotion biases feedback-weighted retrieval toward the current claim **when
  ``feedback_influence`` > 0** (off by default); the ``supersedes`` edge is durable provenance either way.
- **Safety:** ``dry_run=True`` is the default — it reports the detected contradictions + intended
  supersessions without mutating the graph (mirrors ``cleanup_unused_data`` / ``apply_feedback_weights``). A
  hard ``max_pairs`` cap bounds the number of LLM calls per run.

Scope: contradictions are judged over claim-bearing **nodes** (a node's name / description / summary text)
that share a subject — not over raw relationship edges. Bitemporal ``valid_from``/``valid_to`` is intentionally
**out of scope** (the ticket's "to start" path): supersession is encoded purely via the ``supersedes`` edge +
weight demotion.
"""

from typing import Any, Awaitable, Callable, Optional, TypedDict

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger

logger = get_logger("reconcile_memory")

SUPERSEDES = "supersedes"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_DEMOTE_FACTOR = 0.5  # the superseded node's feedback_weight is multiplied by this
DEFAULT_MAX_PAIRS = 24  # hard cap on LLM contradiction checks per run (cost guardrail)
DEFAULT_FEEDBACK_WEIGHT = 0.5  # matches DataPoint.feedback_weight default
_FEEDBACK_DECIMALS = 4
PREFER_RECENCY = "recency"
PREFER_FEEDBACK = "feedback"

_JUDGE_SYSTEM_PROMPT = (
    "You decide whether two knowledge-graph claims about the SAME subject CONTRADICT — they cannot both be "
    "true at the same time (a direct conflict, e.g. 'reports to Bob' vs 'reports to Carol'; 'uses Redis' vs "
    "'uses Memcached'). Complementary facts, layered facts, or claims that merely add detail are NOT "
    "contradictions. When unsure, answer contradicts=false. Return contradicts (bool), confidence (0..1), "
    "and a one-sentence reason."
)


class ContradictionVerdict(BaseModel):
    """Structured verdict from the LLM: do two claims contradict, and how confident is the model."""

    contradicts: bool = Field(
        default=False,
        description="True only if the two claims cannot both be true about the same subject.",
    )
    confidence: float = Field(default=0.0, description="Confidence 0..1 that they contradict.")
    reason: str = Field(default="", description="One short sentence naming the contradiction.")


# --------------------------------------------------------------------------------------
# Pure core (no I/O — unit-tested in isolation)
# --------------------------------------------------------------------------------------
def _node_text(props: dict) -> str:
    """Best-effort textual content of a node for contradiction judging: name + any description/summary text.
    Returns "" for nodes with no usable text (structural/plumbing nodes) so they are skipped."""
    if not isinstance(props, dict):
        return ""
    parts = []
    for key in ("name", "description", "text", "summary", "content"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " — ".join(dict.fromkeys(parts))  # de-dupe, preserve order


def _node_type(props: dict):
    metadata = props.get("metadata") if isinstance(props.get("metadata"), dict) else {}
    return props.get("type") or props.get("node_type") or metadata.get("type")


def candidate_pairs(scannable_ids, edges, max_pairs: int) -> list:
    """Pairs of scannable nodes that SHARE a neighbour (claims about the same subject) — a deterministic
    pre-filter so the LLM only judges plausibly-related claims, never the full O(n^2). The shared neighbour
    can be ANY node, including a *text-less* structural node (e.g. a DocumentChunk / TextSummary that links
    two claims about the same subject); only the pair ENDPOINTS must be scannable. Self-loops are ignored.
    Ordered + de-duped, capped at ``max_pairs``."""
    scannable = set(scannable_ids)
    neighbours: dict = {}
    for edge in edges:
        if not edge or len(edge) < 2:
            continue
        source, target = edge[0], edge[1]
        if source == target:  # ignore self-loops
            continue
        neighbours.setdefault(source, set()).add(target)
        neighbours.setdefault(target, set()).add(source)

    # invert: nodes that share a neighbour are candidate pairs — but only judge SCANNABLE endpoints
    shares: dict = {}
    for nid, nbrs in neighbours.items():
        for nb in nbrs:
            shares.setdefault(nb, set()).add(nid)

    pairs, seen = [], set()
    for members in shares.values():
        members = sorted(m for m in members if m in scannable)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = (members[i], members[j])
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def _ts(meta: dict):
    value = meta.get("updated_at") or meta.get("created_at")
    return value if isinstance(value, (int, float)) else None


def _fw(meta: dict) -> float:
    try:
        return float(meta.get("feedback_weight", DEFAULT_FEEDBACK_WEIGHT))
    except (TypeError, ValueError):
        return DEFAULT_FEEDBACK_WEIGHT


def pick_current(meta_a: dict, meta_b: dict, prefer: str) -> str:
    """Return ``"a"`` or ``"b"`` — the CURRENT claim that supersedes the other, given each node's
    ``{feedback_weight, updated_at, created_at}``. ``recency`` → higher ``updated_at`` (then ``created_at``);
    ``feedback`` → higher ``feedback_weight``. Recency falls back to feedback when timestamps are missing or
    equal, so the decision is always defined."""
    if prefer == PREFER_RECENCY:
        ta, tb = _ts(meta_a), _ts(meta_b)
        if ta is not None and tb is not None and ta != tb:
            return "a" if ta > tb else "b"
        # fall through to feedback when timestamps are missing/equal
    fa, fb = _fw(meta_a), _fw(meta_b)
    if fa != fb:
        return "a" if fa > fb else "b"
    return "a"  # deterministic tie-break


class ReconcileResult(TypedDict):
    scanned: int
    pairs_checked: int
    contradictions: int
    superseded: int
    dry_run: bool
    supersedes: list
    truncated: bool  # True if the candidate scan hit max_pairs (more pairs may be unexamined)


# --------------------------------------------------------------------------------------
# Default LLM judge (the real contradiction decision)
# --------------------------------------------------------------------------------------
async def _default_judge(text_a: str, text_b: str) -> ContradictionVerdict:
    return await LLMGateway.acreate_structured_output(
        text_input=f"Claim A: {text_a}\n\nClaim B: {text_b}",
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        response_model=ContradictionVerdict,
    )


# --------------------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------------------
async def reconcile_memory(
    data: Any = None,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    prefer: str = PREFER_RECENCY,
    demote_factor: float = DEFAULT_DEMOTE_FACTOR,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    protect_node_types: Optional[list] = None,
    dry_run: bool = True,
    judge: Optional[Callable[[str, str], Awaitable[ContradictionVerdict]]] = None,
) -> ReconcileResult:
    """Detect contradictory claims about the same subject, add a ``supersedes`` edge current→stale, and demote
    the stale node's ``feedback_weight``.

    Parameters
    ----------
    data:
        Accepted for memify-pipeline compatibility; ignored (this task walks the whole graph).
    confidence_threshold:
        Minimum LLM confidence (0..1) for a contradiction to be acted on.
    prefer:
        ``"recency"`` (newer ``updated_at``/``created_at`` wins) or ``"feedback"`` (higher ``feedback_weight``).
    demote_factor:
        The superseded node's ``feedback_weight`` is multiplied by this (in [0, 1]).
    max_pairs:
        Hard cap on the number of candidate pairs judged by the LLM per run (cost guardrail).
    protect_node_types:
        Node types never considered (e.g. structural / EntityType nodes).
    dry_run:
        When True (default), report detected contradictions + intended supersessions without mutating.
    judge:
        Async ``(text_a, text_b) -> ContradictionVerdict``. Defaults to the real ``LLMGateway`` call;
        injectable so tests run deterministically and offline.
    """
    judge = judge or _default_judge
    protected = set(protect_node_types or [])
    demote_factor = min(1.0, max(0.0, float(demote_factor)))

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    # scannable nodes: have usable text and are not a protected type
    props_by_id: dict = {}
    for node_id, props in nodes:
        props = props if isinstance(props, dict) else {}
        if _node_type(props) in protected:
            continue
        if _node_text(props):
            props_by_id[node_id] = props

    # AUTHORITATIVE current weights via the graph engine (same method apply_feedback_weights uses),
    # rather than trusting the serialized property blob. Adapters that don't implement feedback weights
    # (e.g. postgres/neptune raise NotImplementedError) fall back to the serialized value so a dry_run still
    # reports cleanly; an actual (non-dry_run) demotion still requires a feedback-weight-capable adapter.
    try:
        stored_weights = await graph_engine.get_node_feedback_weights(list(props_by_id.keys()))
    except NotImplementedError:
        stored_weights = {
            nid: props.get("feedback_weight", DEFAULT_FEEDBACK_WEIGHT)
            for nid, props in props_by_id.items()
        }

    def meta(node_id: str) -> dict:
        props = props_by_id[node_id]
        return {
            "feedback_weight": stored_weights.get(
                node_id, props.get("feedback_weight", DEFAULT_FEEDBACK_WEIGHT)
            ),
            "updated_at": props.get("updated_at"),
            "created_at": props.get("created_at"),
        }

    pairs = candidate_pairs(list(props_by_id.keys()), edges, max_pairs)

    supersedes_out: list = []
    edge_writes: list = []
    weight_updates: dict = {}
    resolved_stale: set = set()  # supersede each stale node once (first/strongest wins)
    contradictions = 0

    for a, b in pairs:
        try:
            verdict = await judge(_node_text(props_by_id[a]), _node_text(props_by_id[b]))
        except Exception as exc:  # a single bad judgment must not abort the whole run
            logger.warning("reconcile_memory: judge failed for (%s, %s): %r", a, b, exc)
            continue
        if not getattr(verdict, "contradicts", False):
            continue
        confidence = float(getattr(verdict, "confidence", 0.0) or 0.0)
        if confidence < confidence_threshold:
            continue
        contradictions += 1

        winner = pick_current(meta(a), meta(b), prefer)
        current, stale = (a, b) if winner == "a" else (b, a)
        if stale in resolved_stale:
            continue
        resolved_stale.add(stale)

        reason = str(getattr(verdict, "reason", "") or "")[:200]
        supersedes_out.append(
            {"current": current, "stale": stale, "confidence": confidence, "reason": reason}
        )
        edge_writes.append(
            (
                current,
                stale,
                SUPERSEDES,
                {
                    "source_node_id": current,
                    "target_node_id": stale,
                    "relationship_name": SUPERSEDES,
                    "confidence": confidence,
                    "reason": reason,
                },
            )
        )
        stale_weight = float(stored_weights.get(stale, DEFAULT_FEEDBACK_WEIGHT))
        weight_updates[stale] = round(max(0.0, stale_weight * demote_factor), _FEEDBACK_DECIMALS)

    if not dry_run:
        for source, target, relationship, edge_props in edge_writes:
            await graph_engine.add_edge(source, target, relationship, edge_props)
        if weight_updates:
            await graph_engine.set_node_feedback_weights(weight_updates)

    # the candidate scan stops at max_pairs; signal it so a caller never reads contradictions=0
    # as "graph is clean" when it was really "we stopped early" on a large graph.
    truncated = len(pairs) >= max_pairs
    logger.info(
        "reconcile_memory: scanned=%d pairs=%d contradictions=%d superseded=%d dry_run=%s truncated=%s",
        len(props_by_id),
        len(pairs),
        contradictions,
        len(edge_writes),
        dry_run,
        truncated,
    )
    return {
        "scanned": len(props_by_id),
        "pairs_checked": len(pairs),
        "contradictions": contradictions,
        "superseded": len(edge_writes),
        "dry_run": dry_run,
        "supersedes": supersedes_out,
        "truncated": truncated,
    }
