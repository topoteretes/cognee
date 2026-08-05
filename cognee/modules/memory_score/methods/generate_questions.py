"""Synthetic question generation for the memory accuracy score.

Step 3 of the memory-score pipeline: for every topic produced by ``build_topics``,
pull the real text of one or more chunks that belong to that topic and ask the LLM
for (question, expected_answer) pairs grounded in that text. The expected answer
becomes the golden answer handed to the correctness judge, so it must be provable
from the chunk alone — hence the grounding, and hence the prompt's instruction to
return fewer pairs rather than invent one.

Cost shape: exactly one LLM call per topic (the requested pairs are batched into a
single structured response), never one call per question.

Everything here is best-effort per topic: a topic whose nodes yield no usable text,
or whose LLM call fails, is skipped with a warning instead of failing the run.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.memory_score.methods.build_topics import Topic, TopicPlan
from cognee.shared.logging_utils import get_logger

logger = get_logger("memory_score.generate_questions")

SYSTEM_PROMPT_FILE = "memory_score_question_generation_system.txt"
USER_PROMPT_FILE = "memory_score_question_generation_user.txt"

# Graph node type that carries the ingested text a question can be grounded in.
CHUNK_NODE_TYPE = "DocumentChunk"

# How many of a topic's nodes we look at. Fetching a node materializes its full
# `text` property, so this is a cost guard, not a semantic limit.
MAX_NODES_PER_TOPIC = 60
# Topics made purely of entities have no text of their own; their chunk is one hop
# away. We probe only a few of them so the fallback stays cheap.
NEIGHBOUR_PROBE_LIMIT = 5
# Grounding context sent to the LLM for one topic.
MAX_CHUNKS_PER_PROMPT = 3
MAX_CONTEXT_CHARS = 6000
MIN_TEXT_CHARS = 40

# Base weight every topic carries in the allocation, regardless of real traffic.
REAL_TRAFFIC_WEIGHT_BASE = 1


@dataclass
class GeneratedQuestion:
    """One synthetic (question, expected_answer) pair, tagged with its topic."""

    text: str
    expected_answer: str
    topic: str


class QuestionPair(BaseModel):
    """A single LLM-authored pair, grounded in the source text it was shown."""

    question: str = Field(description="Question answerable from the source text alone.")
    expected_answer: str = Field(description="The answer as stated by the source text.")


class QuestionPairList(BaseModel):
    """Structured question-generation response: the pairs for one topic (possibly empty)."""

    questions: list[QuestionPair] = Field(default_factory=list)


def _allocate_counts(topics: list[Topic], target_count: int) -> list[int]:
    """Split ``target_count`` questions across ``topics``, favouring real traffic.

    Weighting: ``REAL_TRAFFIC_WEIGHT_BASE + real_question_count``. A topic nobody has
    ever asked about still gets the base weight (the memory can be wrong there too),
    and every real question that landed in a topic adds one unit — so a topic hit by
    9 real questions is weighted 10x a topic hit by none. Every topic then keeps a
    floor of one question so no topic goes unmeasured, and the remaining budget is
    handed out by weight with largest-remainder rounding so the parts sum to exactly
    ``target_count``.

    ``target_count`` is a STRICT CEILING, never a target the floor is allowed to
    overshoot: it is the caller's spend authorisation, and the run is billed per
    question. When the budget cannot cover one question per topic, the per-topic
    floor is what gives — the most-asked-about topics keep their question and the
    rest get zero (skipped by :func:`generate_questions`, which also spares their
    graph reads). Measuring fewer topics than asked for is a legible outcome;
    quietly generating 5 questions for a caller who authorised 2 is not.
    """
    topic_count = len(topics)
    if topic_count == 0 or target_count <= 0:
        return [0] * topic_count

    weights = [REAL_TRAFFIC_WEIGHT_BASE + max(0, topic.real_question_count) for topic in topics]

    if target_count <= topic_count:
        # Ties break on the lower index, as in the largest-remainder pass below.
        order = sorted(range(topic_count), key=lambda index: (-weights[index], index))
        counts = [0] * topic_count
        for index in order[:target_count]:
            counts[index] = 1
        return counts

    remaining = target_count - topic_count
    total_weight = sum(weights)

    shares = [remaining * weight / total_weight for weight in weights]
    counts = [int(share) for share in shares]

    # Largest-remainder pass; ties break on the lower index so the split is stable.
    leftover = remaining - sum(counts)
    if leftover > 0:
        order = sorted(
            range(topic_count), key=lambda index: (-(shares[index] - counts[index]), index)
        )
        for index in order[:leftover]:
            counts[index] += 1

    return [1 + count for count in counts]


def _node_text(node) -> str | None:
    """Return the node's usable text, or None when it carries none worth grounding on."""
    if not isinstance(node, dict):
        return None

    text = node.get("text")
    if not isinstance(text, str):
        return None

    text = text.strip()
    return text if len(text) >= MIN_TEXT_CHARS else None


def _split_by_type(nodes: list) -> tuple[list[str], list[str]]:
    """Split node texts into chunk texts and any other text-bearing node's text."""
    chunk_texts: list[str] = []
    other_texts: list[str] = []

    for node in nodes:
        text = _node_text(node)
        if text is None:
            continue
        if isinstance(node, dict) and node.get("type") == CHUNK_NODE_TYPE:
            chunk_texts.append(text)
        else:
            other_texts.append(text)

    return chunk_texts, other_texts


async def _neighbour_chunk_texts(graph_engine, node_ids: list[str]) -> list[str]:
    """Look one hop out for chunk text, for topics that are entity nodes only."""
    texts: list[str] = []

    for node_id in node_ids[:NEIGHBOUR_PROBE_LIMIT]:
        try:
            neighbours = await graph_engine.get_neighbors(node_id) or []
        except Exception as error:
            logger.warning("memory score: failed to read neighbours of %s: %s", node_id, error)
            continue

        chunk_texts, _ = _split_by_type(neighbours)
        texts.extend(chunk_texts)

        if len(texts) >= MAX_CHUNKS_PER_PROMPT:
            break

    return texts


async def _collect_topic_texts(graph_engine, topic: Topic) -> list[str]:
    """Collect the text a topic's questions can be grounded in, best chunk first."""
    node_ids = [str(node_id) for node_id in (topic.node_ids or []) if node_id]
    if not node_ids:
        return []

    try:
        nodes = await graph_engine.get_nodes(node_ids[:MAX_NODES_PER_TOPIC]) or []
    except Exception as error:
        logger.warning("memory score: failed to read nodes for topic '%s': %s", topic.label, error)
        return []

    chunk_texts, other_texts = _split_by_type(nodes)
    if chunk_texts:
        return chunk_texts

    neighbour_texts = await _neighbour_chunk_texts(graph_engine, node_ids)
    if neighbour_texts:
        return neighbour_texts

    # Summaries and other text-bearing nodes are a weaker but still real grounding.
    return other_texts


def _build_context(texts: list[str]) -> str:
    """Join a few of the collected texts into one bounded grounding context."""
    selected: list[str] = []
    budget = MAX_CONTEXT_CHARS

    for text in texts[:MAX_CHUNKS_PER_PROMPT]:
        if budget <= 0:
            break
        selected.append(text[:budget])
        budget -= len(selected[-1])

    return "\n\n---\n\n".join(selected)


async def _generate_for_topic(topic: Topic, count: int, context: str) -> list[GeneratedQuestion]:
    """Ask the LLM for ``count`` pairs grounded in ``context``, in a single call."""
    # Only the PROMPT gets a fallback name: every pair is tagged with the plan's
    # own ``topic.label``, which is what the per-topic aggregate keys its rows by.
    # Tagging with a substituted label would file the questions under a topic the
    # aggregate does not have, and they would appear in no per-topic row at all.
    prompt_label = topic.label or "cluster"

    user_prompt = render_prompt(
        USER_PROMPT_FILE, {"topic": prompt_label, "source_text": context, "count": count}
    )
    system_prompt = read_query_prompt(SYSTEM_PROMPT_FILE)
    if not system_prompt:
        logger.warning("memory score: missing prompt file %s", SYSTEM_PROMPT_FILE)
        return []

    result = await LLMGateway.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=QuestionPairList,
    )

    questions: list[GeneratedQuestion] = []
    seen: set[str] = set()

    for pair in result.questions:
        text = (pair.question or "").strip()
        expected_answer = (pair.expected_answer or "").strip()
        if not text or not expected_answer:
            continue

        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)

        questions.append(
            GeneratedQuestion(text=text, expected_answer=expected_answer, topic=topic.label)
        )

        # The model was asked for `count`; never accept more than we budgeted.
        if len(questions) >= count:
            break

    return questions


async def generate_questions(topic_plan: TopicPlan, target_count: int) -> list[GeneratedQuestion]:
    """Generate synthetic (question, expected_answer) pairs for a topic plan.

    Args:
        topic_plan: The plan from ``build_topics``. A plan below the data floor, or
            with no topics, generates nothing.
        target_count: Strict upper bound on questions across all topics. It is
            split by ``_allocate_counts`` (real-traffic weighted, one-per-topic floor
            where the budget allows).

    Returns:
        The generated pairs, grouped topic by topic. Never more than
        ``target_count``. Topics whose nodes yield no usable text — and topics whose
        LLM call fails — are skipped, so the result can be shorter and can
        legitimately be empty.
    """
    if topic_plan is None or topic_plan.below_data_floor:
        return []

    topics = [topic for topic in (topic_plan.topics or []) if topic is not None]
    if not topics or target_count <= 0:
        return []

    counts = _allocate_counts(topics, target_count)
    graph_engine = await get_graph_engine()

    generated: list[GeneratedQuestion] = []

    for topic, count in zip(topics, counts):
        # Zero budget: the target could not cover every topic. Skipped before the
        # graph reads, since nothing would be generated from them anyway.
        if count <= 0:
            continue

        texts = await _collect_topic_texts(graph_engine, topic)
        context = _build_context(texts)
        if not context:
            logger.warning(
                "memory score: topic '%s' has no usable chunk text, skipping generation",
                topic.label,
            )
            continue

        try:
            generated.extend(await _generate_for_topic(topic, count, context))
        except Exception as error:
            logger.warning(
                "memory score: question generation failed for topic '%s': %s",
                topic.label,
                error,
                exc_info=True,
            )

    logger.info(
        "memory score: generated %d synthetic question(s) across %d topic(s)",
        len(generated),
        len(topics),
    )

    return generated
