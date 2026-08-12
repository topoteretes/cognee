"""Propose new topics from dense clusters of sink questions.

Phase 2 steps 8-10 of the recall-coverage spec. The sink is the honest answer for
a question the taxonomy cannot place; a *dense group* of such questions is
something else — evidence of a topic the owner has not accepted yet. This module
turns those groups into ``pending`` suggestion rows and stops there: the topic id
is minted on accept, which is what makes accepted topic ids stable across runs.

Three decisions worth stating outright, because each has a tempting wrong version:

* **Not k-means.** ``cognee/modules/visualization/semantic_clusters.py``'s
  ``compute_clusters`` is deliberately unused here: k-means returns exactly ``k``
  partitions whether or not any of them is dense, so it would confidently propose
  ``k`` topics out of scattered noise. The requirement is literally "a dense
  cluster", so this reuses the same single-link grouping dedup uses
  (:func:`cognee.modules.recall_coverage.dedup.group_by_similarity`) at the looser
  ``sink_cluster_threshold`` — looser because a suggestion is a theme, not a
  duplicate.
* **``cohesion`` orders candidates and is never scored.** It is the mean
  intra-cluster cosine: useful for "propose the tightest five first", meaningless
  as a quality number about memory, and deliberately absent from every aggregate.
* **The re-proposal guard is owner-scoped, spanning agent labels.** Dismissing a
  suggestion is a statement about the owner's taxonomy, not about one tool, so a
  suggestion dismissed on the Codex run must not reappear on the Claude Code run.
  ``agent_label`` on a suggestion is provenance only. A *pending* suggestion from
  another run is not consulted: it has not been decided, so re-surfacing the same
  theme is correct rather than noise.

One LLM call per surviving cluster, and only for survivors — filtering, capping
and the guard all run first, so a run that proposes nothing costs nothing.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, create_model

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.recall_coverage.dedup import group_by_similarity
from cognee.modules.recall_coverage.embedding import EmbeddingFingerprint
from cognee.modules.recall_coverage.repository import (
    SuggestionDraft,
    SuggestionRecord,
    create_topic_suggestions,
    load_settled_suggestions,
)
from cognee.modules.recall_coverage.types import CoverageParams
from cognee.root_dir import get_absolute_path
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")

PROMPT_DIRECTORY = "./modules/recall_coverage/prompts"
LABEL_INPUT_PROMPT = "recall_coverage_topic_label_input.txt"
LABEL_SYSTEM_PROMPT = "recall_coverage_topic_label_system.txt"


@dataclass(frozen=True)
class SinkCluster:
    """A dense group of sink questions, before it has a label.

    ``member_indices`` index into the sink question list the cluster was built
    from, so the caller can recover the texts (and, after acceptance, know which
    questions motivated the topic).
    """

    member_indices: tuple[int, ...]
    # L2-normalized mean of the members, so a dot product against a normalized
    # question vector is already the cosine similarity — the same contract the
    # ``centroid`` column carries.
    centroid: tuple[float, ...]
    # Mean intra-cluster cosine. Orders candidates; never scored.
    cohesion: float

    @property
    def question_count(self) -> int:
        return len(self.member_indices)


def cluster_centroid(vectors: np.ndarray) -> tuple[float, ...]:
    """The L2-normalized mean of ``vectors``.

    A degenerate cluster whose mean is the zero vector (members pointing in
    opposite directions, or every member's embedding having failed) stays zero:
    cosine-similar to nothing, so it can never be matched by the re-proposal
    guard or by assignment, which is the correct behaviour for a centroid that
    means nothing.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return ()

    mean = np.asarray(vectors, dtype=float).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        return tuple(float(value) for value in mean)
    return tuple(float(value) for value in mean / norm)


def cluster_cohesion(vectors: np.ndarray) -> float:
    """Mean pairwise cosine similarity inside a cluster.

    **Ordering only.** It says how tight a proposed theme is, not how well memory
    answers it, and no aggregate reads it. A single-member cluster has no pair to
    average, and is reported as ``1.0`` (a cluster is trivially similar to
    itself); such clusters are dropped by ``min_questions_per_topic`` long before
    this matters.
    """
    count = int(vectors.shape[0]) if vectors.ndim == 2 else 0
    if count < 2:
        return 1.0

    similarity = np.asarray(vectors, dtype=float) @ np.asarray(vectors, dtype=float).T
    upper = np.triu_indices(count, k=1)
    return float(np.mean(similarity[upper]))


def cluster_sink_questions(
    vectors: np.ndarray,
    *,
    sink_cluster_threshold: float,
    min_questions_per_topic: int,
    max_suggestions_per_run: int,
) -> list[SinkCluster]:
    """Group the sink questions, keep the dense ones, order them by cohesion.

    Order of operations follows the spec exactly: cluster, drop clusters below
    ``min_questions_per_topic``, sort by cohesion, then cap at
    ``max_suggestions_per_run``. The cap is applied **before** the re-proposal
    guard, so a run whose tightest clusters have all been dismissed before can
    surface fewer than the cap rather than reaching further down the list — the
    cap is a "don't show me ten" limit on what the run considered, not a quota to
    fill.
    """
    count = int(vectors.shape[0]) if vectors.ndim == 2 else 0
    if count == 0:
        return []

    groups, _comparisons = group_by_similarity(vectors, sink_cluster_threshold)

    clusters: list[SinkCluster] = []
    for group in groups:
        if len(group) < min_questions_per_topic:
            continue
        members = np.asarray(vectors, dtype=float)[group]
        clusters.append(
            SinkCluster(
                member_indices=tuple(group),
                centroid=cluster_centroid(members),
                cohesion=cluster_cohesion(members),
            )
        )

    # Tightest first; the member count breaks ties so the order is deterministic
    # for two equally tight clusters.
    clusters.sort(key=lambda cluster: (-cluster.cohesion, -cluster.question_count))

    if max_suggestions_per_run >= 0:
        clusters = clusters[:max_suggestions_per_run]
    return clusters


def drop_reproposed(
    clusters: Sequence[SinkCluster],
    settled: Sequence[SuggestionRecord],
    *,
    fingerprint: EmbeddingFingerprint,
    suggestion_dedup_threshold: float,
) -> list[SinkCluster]:
    """Drop candidates that repeat a suggestion the owner already accepted or dismissed.

    Without this, dismissing a suggestion does not stick: the same dense cluster
    is in the sink next run and would be proposed again forever.

    A settled suggestion stored under a different embedding fingerprint is
    skipped with a warning rather than failing the run. Unlike a topic centroid —
    which is scored against and therefore must fail (see
    :func:`cognee.modules.recall_coverage.assign.require_matching_fingerprint`) —
    a stale suggestion only weakens this filter, and the visible consequence is a
    re-proposal the owner can dismiss again, not a wrong number.
    """
    if not clusters or not settled:
        return list(clusters)

    usable: list[SuggestionRecord] = []
    for suggestion in settled:
        if suggestion.embedding_model != fingerprint.model or (
            fingerprint.dimensions > 0 and suggestion.embedding_dimensions != fingerprint.dimensions
        ):
            logger.warning(
                "recall_coverage: ignoring %s suggestion %s in the re-proposal guard; it was "
                "embedded with %r/%s, the live engine is %r/%s",
                suggestion.status,
                suggestion.id,
                suggestion.embedding_model,
                suggestion.embedding_dimensions,
                fingerprint.model,
                fingerprint.dimensions,
            )
            continue
        usable.append(suggestion)

    if not usable:
        return list(clusters)

    width = max(len(cluster.centroid) for cluster in clusters)
    settled_rows = [
        list(suggestion.centroid) for suggestion in usable if len(suggestion.centroid) == width
    ]
    if not settled_rows:
        return list(clusters)

    settled_matrix = np.asarray(settled_rows, dtype=float)
    candidate_matrix = np.asarray([list(cluster.centroid) for cluster in clusters], dtype=float)
    similarity = candidate_matrix @ settled_matrix.T

    kept: list[SinkCluster] = []
    for index, cluster in enumerate(clusters):
        if float(np.max(similarity[index])) >= suggestion_dedup_threshold:
            logger.debug(
                "recall_coverage: dropping a %s-question suggestion candidate already settled",
                cluster.question_count,
            )
            continue
        kept.append(cluster)
    return kept


@lru_cache
def topic_label_model(max_chars: int) -> type[BaseModel]:
    """The structured-output model for one topic label.

    Built per ``topic_label_max_chars`` rather than declared with a literal
    ``max_length``: the limit is a configuration parameter like every other
    threshold here, and baking a number into the class would make it the one
    magic number in the module. Cached because the class identity is what the
    structured-output framework keys its schema on.
    """
    return create_model(
        "TopicLabel",
        label=(
            str,
            Field(
                max_length=max_chars,
                description="A short noun-phrase naming what these questions are about.",
            ),
        ),
    )


def _label_prompts(texts: Sequence[str], *, max_chars: int) -> tuple[str, str]:
    """Render the user prompt and read the system prompt for label generation.

    ``read_query_prompt`` returns ``None`` for a missing file instead of raising,
    which would send ``system_prompt=None`` to the provider and produce a label
    from no instructions at all. A missing prompt file is a packaging bug, so it
    is turned into a loud failure here.
    """
    base_directory = get_absolute_path(PROMPT_DIRECTORY)
    text_input = render_prompt(
        LABEL_INPUT_PROMPT,
        {"questions": list(texts), "max_chars": max_chars},
        base_directory=base_directory,
    )
    system_prompt = read_query_prompt(LABEL_SYSTEM_PROMPT, base_directory=base_directory)
    if not system_prompt:
        raise FileNotFoundError(
            f"recall-coverage topic label prompt {LABEL_SYSTEM_PROMPT} is missing from "
            f"{base_directory}"
        )
    return text_input, system_prompt


async def generate_topic_label(texts: Sequence[str], *, topic_label_max_chars: int) -> str:
    """One LLM call naming what a cluster of questions is about.

    The returned label is truncated defensively: ``max_length`` in the response
    model is a request to the provider, and a provider that ignores it must not
    be able to write a paragraph into a column the UI renders as a chip.
    """
    text_input, system_prompt = _label_prompts(texts, max_chars=topic_label_max_chars)

    response = await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,
        response_model=topic_label_model(topic_label_max_chars),
    )

    label = str(getattr(response, "label", "") or "").strip()
    return label[:topic_label_max_chars]


async def suggest_topics(
    owner_id: UUID,
    texts: Sequence[str],
    vectors: np.ndarray,
    *,
    params: CoverageParams,
    fingerprint: EmbeddingFingerprint,
    run_id: Optional[UUID] = None,
    agent_label: Optional[str] = None,
    settled: Optional[Sequence[SuggestionRecord]] = None,
) -> list[SuggestionRecord]:
    """Turn this run's sink questions into ``pending`` topic suggestions.

    ``texts`` and ``vectors`` are the sink questions' canonical texts and their
    canonical (already normalized) vectors, index-aligned — see
    :func:`cognee.modules.recall_coverage.assign.canonical_matrix`. ``settled`` is
    injectable so a caller that already loaded the owner's decided suggestions
    does not read them twice.

    A cluster whose label generation fails is dropped rather than stored with a
    placeholder: an unlabelled suggestion is not reviewable, and suggestions are
    advisory, so losing one must never fail a run that has already judged
    everything.
    """
    clusters = cluster_sink_questions(
        vectors,
        sink_cluster_threshold=params.sink_cluster_threshold,
        min_questions_per_topic=params.min_questions_per_topic,
        max_suggestions_per_run=params.max_suggestions_per_run,
    )
    if not clusters:
        return []

    if settled is None:
        settled = await load_settled_suggestions(owner_id)

    clusters = drop_reproposed(
        clusters,
        settled,
        fingerprint=fingerprint,
        suggestion_dedup_threshold=params.suggestion_dedup_threshold,
    )
    if not clusters:
        return []

    drafts: list[SuggestionDraft] = []
    for cluster in clusters:
        member_texts = [texts[index] for index in cluster.member_indices if index < len(texts)]
        try:
            label = await generate_topic_label(
                member_texts, topic_label_max_chars=params.topic_label_max_chars
            )
        except Exception as error:
            logger.warning(
                "recall_coverage: dropping a %s-question suggestion candidate; "
                "label generation failed: %s",
                cluster.question_count,
                error,
            )
            continue

        if not label:
            logger.warning("recall_coverage: dropping a suggestion candidate with an empty label")
            continue

        drafts.append(
            SuggestionDraft(
                owner_id=owner_id,
                label=label,
                centroid=cluster.centroid,
                embedding_model=fingerprint.model,
                embedding_dimensions=fingerprint.dimensions or len(cluster.centroid),
                question_count=cluster.question_count,
                cohesion=cluster.cohesion,
                agent_label=agent_label,
                run_id=run_id,
            )
        )

    if not drafts:
        return []

    return await create_topic_suggestions(drafts)


__all__ = [
    "SinkCluster",
    "cluster_centroid",
    "cluster_cohesion",
    "cluster_sink_questions",
    "drop_reproposed",
    "generate_topic_label",
    "suggest_topics",
    "topic_label_model",
]
