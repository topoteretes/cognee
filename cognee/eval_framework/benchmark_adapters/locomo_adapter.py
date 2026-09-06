"""LoCoMo benchmark adapter — very long-term conversational memory.

Dataset: https://github.com/snap-research/locomo (``data/locomo10.json``)
Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (Maharana et al., 2024)

``locomo10.json`` is a list of 10 conversation records. Each record has:

- ``sample_id``: e.g. ``"conv-26"``.
- ``conversation``: ``speaker_a``, ``speaker_b``, and for every session ``n`` a
  ``session_<n>`` list of turns plus a ``session_<n>_date_time`` string such as
  ``"1:56 pm on 8 May, 2023"``. A turn is ``{speaker, dia_id, text}`` with optional
  ``img_url`` / ``blip_caption`` / ``query`` when the speaker shared a photo.
- ``qa``: list of ``{question, answer | adversarial_answer, evidence, category}``.
  ``evidence`` is a stringified list of ``dia_id`` values (``"['D2:8']"``).

Category semantics (verified on the data: mean evidence count and question shape):

| category | name          | notes                                              |
| -------- | ------------- | -------------------------------------------------- |
| 1        | multi_hop     | ~3 evidence turns per question                     |
| 2        | temporal      | ~80% "when / how long ago" questions               |
| 3        | open_domain   | inference over the conversation + world knowledge  |
| 4        | single_hop    | one evidence turn                                  |
| 5        | adversarial   | unanswerable; gold lives in ``adversarial_answer`` |
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import BaseBenchmarkAdapter
from cognee.shared.logging_utils import get_logger

logger = get_logger()

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_FILENAME = "locomo10.json"

CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
ADVERSARIAL_CATEGORY = 5
# The answer the adapter records for adversarial (unanswerable) questions. The
# dataset's ``adversarial_answer`` is the *distractor* the model should not give,
# so it is kept under its own key and the gold answer becomes an abstention.
ADVERSARIAL_GOLD_ANSWER = "The conversation does not contain this information."


@dataclass
class LocomoTurn:
    speaker: str
    dia_id: str
    text: str
    blip_caption: str | None = None
    img_url: str | None = None


@dataclass
class LocomoSession:
    index: int  # 1-based, as in the dataset keys
    date_time: str
    turns: list[LocomoTurn] = field(default_factory=list)


@dataclass
class LocomoConversation:
    conversation_index: int
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[LocomoSession]
    qa: list[dict[str, Any]]

    @property
    def turn_count(self) -> int:
        return sum(len(session.turns) for session in self.sessions)

    def turn_lookup(self) -> dict[str, tuple[LocomoSession, LocomoTurn]]:
        lookup: dict[str, tuple[LocomoSession, LocomoTurn]] = {}
        for session in self.sessions:
            for turn in session.turns:
                lookup[turn.dia_id] = (session, turn)
        return lookup


def category_name(category: int | str) -> str:
    try:
        return CATEGORY_NAMES[int(category)]
    except (KeyError, TypeError, ValueError):
        return f"category_{category}"


def parse_evidence(evidence: Any) -> list[str]:
    """``evidence`` is usually a stringified Python list; tolerate real lists too."""
    if evidence is None:
        return []
    if isinstance(evidence, list):
        return [str(item) for item in evidence]
    if isinstance(evidence, str):
        try:
            parsed = ast.literal_eval(evidence)
        except (ValueError, SyntaxError):
            return [evidence] if evidence.strip() else []
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(evidence)]


def _session_keys(conversation: dict[str, Any]) -> list[int]:
    indices = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("_date_time"):
            suffix = key[len("session_") :]
            if suffix.isdigit():
                indices.append(int(suffix))
    return sorted(indices)


def parse_conversation(record: dict[str, Any], conversation_index: int) -> LocomoConversation:
    conversation = record["conversation"]
    sessions: list[LocomoSession] = []
    for index in _session_keys(conversation):
        raw_turns = conversation.get(f"session_{index}") or []
        turns = [
            LocomoTurn(
                speaker=str(turn.get("speaker", "")),
                dia_id=str(turn.get("dia_id", f"D{index}:{position + 1}")),
                text=str(turn.get("text", "")),
                blip_caption=turn.get("blip_caption"),
                img_url=turn.get("img_url"),
            )
            for position, turn in enumerate(raw_turns)
        ]
        sessions.append(
            LocomoSession(
                index=index,
                date_time=str(conversation.get(f"session_{index}_date_time", "unknown date")),
                turns=turns,
            )
        )

    return LocomoConversation(
        conversation_index=conversation_index,
        sample_id=str(record.get("sample_id", f"conv-{conversation_index}")),
        speaker_a=str(conversation.get("speaker_a", "Speaker A")),
        speaker_b=str(conversation.get("speaker_b", "Speaker B")),
        sessions=sessions,
        qa=list(record.get("qa") or []),
    )


def format_turn(turn: LocomoTurn) -> str:
    line = f"{turn.speaker}: {turn.text}".rstrip()
    if turn.blip_caption:
        line += f" [shares a photo: {turn.blip_caption}]"
    return line


def flatten_conversation(conversation: LocomoConversation) -> str:
    """Single-document transcript used by the generic ``cognee eval`` corpus path."""
    lines = [
        (
            f"Conversation between {conversation.speaker_a} and {conversation.speaker_b} "
            f"({conversation.sample_id})."
        )
    ]
    for session in conversation.sessions:
        lines.append("")
        lines.append(f"--- Session {session.index} — {session.date_time} ---")
        for turn in session.turns:
            lines.append(format_turn(turn))
    return "\n".join(lines)


def build_question_records(
    conversation: LocomoConversation,
    *,
    include_adversarial: bool = True,
    load_golden_context: bool = False,
    max_session_index: int | None = None,
) -> list[dict[str, Any]]:
    """Turn a conversation's ``qa`` list into eval-framework question dicts.

    When ``max_session_index`` is given (the conversation was truncated), questions whose
    evidence references a later session are dropped so a partial ingestion is still scored
    against answerable questions only.
    """
    lookup = conversation.turn_lookup() if load_golden_context else {}
    records: list[dict[str, Any]] = []

    for position, qa in enumerate(conversation.qa):
        try:
            category = int(qa.get("category", -1))
        except (TypeError, ValueError):
            category = -1

        if category == ADVERSARIAL_CATEGORY and not include_adversarial:
            continue

        evidence = parse_evidence(qa.get("evidence"))
        if max_session_index is not None:
            referenced_sessions = []
            for dia_id in evidence:
                head = dia_id.split(":")[0].lstrip("D")
                if head.isdigit():
                    referenced_sessions.append(int(head))
            if referenced_sessions and max(referenced_sessions) > max_session_index:
                continue

        if category == ADVERSARIAL_CATEGORY:
            gold = ADVERSARIAL_GOLD_ANSWER
        else:
            gold = qa.get("answer")
            gold = "" if gold is None else str(gold)

        record: dict[str, Any] = {
            "question": str(qa.get("question", "")),
            "answer": gold,
            "question_type": category_name(category),
            "category": category,
            "evidence": evidence,
            "conversation_id": conversation.sample_id,
            "conversation_index": conversation.conversation_index,
            "question_idx": position,
        }
        if qa.get("adversarial_answer") is not None:
            record["adversarial_answer"] = str(qa["adversarial_answer"])
        if category == ADVERSARIAL_CATEGORY and qa.get("answer") is not None:
            record["dataset_answer"] = str(qa["answer"])

        if load_golden_context:
            golden_lines = []
            for dia_id in evidence:
                hit = lookup.get(dia_id)
                if hit is None:
                    continue
                session, turn = hit
                golden_lines.append(
                    f"[Session {session.index}, {session.date_time}] {format_turn(turn)}"
                )
            if golden_lines:
                record["golden_context"] = "\n".join(golden_lines)

        records.append(record)

    return records


class LocomoAdapter(BaseBenchmarkAdapter):
    """Adapter for the LoCoMo long-term conversational memory benchmark.

    Args:
        conversation_index: Which conversation to load (0-indexed). ``None`` loads all ten;
            the corpus then has one document per conversation and questions carry
            ``conversation_index`` so a caller can group them.
        max_sessions: Truncate each conversation to its first N sessions (fast local runs).
            Questions whose evidence lies beyond the cut are dropped.
        include_adversarial: Keep category-5 (unanswerable) questions. The mem0 paper drops
            them; the LoCoMo paper keeps them. Default keeps them.
        data_path: Local path to ``locomo10.json``. Defaults to ``LOCOMO_DATA_PATH`` env or
            ``./locomo10.json``; downloaded from the official repository when missing.
    """

    dataset_info = {"filename": LOCOMO_FILENAME, "url": LOCOMO_URL}

    def __init__(
        self,
        conversation_index: int | None = None,
        max_sessions: int | None = None,
        include_adversarial: bool = True,
        data_path: str | None = None,
    ):
        self.conversation_index = conversation_index
        self.max_sessions = max_sessions
        self.include_adversarial = include_adversarial
        self.data_path = data_path or os.getenv("LOCOMO_DATA_PATH") or LOCOMO_FILENAME
        self._records: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------ loading
    def _get_raw_records(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records

        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
        else:
            # Imported lazily: requests is not a declared cognee dependency.
            import requests

            logger.info("Downloading LoCoMo dataset from %s", self.dataset_info["url"])
            response = requests.get(self.dataset_info["url"], timeout=120)
            response.raise_for_status()
            records = response.json()
            os.makedirs(os.path.dirname(os.path.abspath(self.data_path)), exist_ok=True)
            with open(self.data_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)

        if not isinstance(records, list):
            raise ValueError("locomo10.json must be a JSON list of conversation records")
        self._records = records
        return records

    def conversation_count(self) -> int:
        return len(self._get_raw_records())

    def load_conversation(self, conversation_index: int) -> LocomoConversation:
        records = self._get_raw_records()
        if conversation_index < 0 or conversation_index >= len(records):
            raise IndexError(
                f"conversation_index={conversation_index} out of range "
                f"(dataset has {len(records)} conversations)"
            )
        conversation = parse_conversation(records[conversation_index], conversation_index)
        if self.max_sessions is not None:
            conversation.sessions = conversation.sessions[: max(0, self.max_sessions)]
        return conversation

    def iter_conversations(self) -> Iterable[LocomoConversation]:
        indices = (
            [self.conversation_index]
            if self.conversation_index is not None
            else range(self.conversation_count())
        )
        for index in indices:
            yield self.load_conversation(index)

    def questions_for(
        self,
        conversation: LocomoConversation,
        *,
        load_golden_context: bool = False,
    ) -> list[dict[str, Any]]:
        max_session_index = None
        if self.max_sessions is not None and conversation.sessions:
            max_session_index = conversation.sessions[-1].index
        return build_question_records(
            conversation,
            include_adversarial=self.include_adversarial,
            load_golden_context=load_golden_context,
            max_session_index=max_session_index,
        )

    # -------------------------------------------------------- adapter interface
    def load_corpus(
        self,
        limit: int | None = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: str | list[str] | list[int] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Return one transcript per selected conversation plus its questions.

        ``limit`` caps the number of questions (after filtering), matching the other
        adapters' ``number_of_samples_in_corpus`` semantics.
        """
        corpus: list[str] = []
        questions: list[dict[str, Any]] = []

        for conversation in self.iter_conversations():
            corpus.append(flatten_conversation(conversation))
            questions.extend(
                self.questions_for(conversation, load_golden_context=load_golden_context)
            )

        if instance_filter is not None:
            questions = self._filter_instances(questions, instance_filter, id_key="question")

        if limit is not None and limit < len(questions):
            questions = questions[:limit]

        logger.info(
            "Loaded LoCoMo: %s conversation(s), %s questions",
            len(corpus),
            len(questions),
        )
        return corpus, questions
