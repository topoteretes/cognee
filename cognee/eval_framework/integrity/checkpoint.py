import json
import os
from typing import Any

from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.status import (
    AnswerStatus,
    answer_status,
    record_id,
    source_question_id,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

JOURNAL_SUFFIX = ".journal.jsonl"


def journal_path(answers_path: str) -> str:
    return answers_path + JOURNAL_SUFFIX


def _drop_torn_tail(path: str) -> None:
    """A crash mid-append leaves a last line without its newline. Cut the file back to the last
    complete line so appends start on a clean line; everything before it is untouched."""
    with open(path, "rb+") as handle:
        content = handle.read()
        if not content or content.endswith(b"\n"):
            return
        keep = content.rfind(b"\n") + 1
        logger.warning(
            "Answer journal %s ends with a partial record (%d bytes); dropping it, that "
            "question will be asked again",
            path,
            len(content) - keep,
        )
        handle.truncate(keep)
        handle.flush()
        os.fsync(handle.fileno())


def _read_journal(path: str) -> list[dict[str, Any]]:
    """Read complete journal lines. A bad line is corruption, not a crash artifact (torn tails
    are removed by ``_drop_torn_tail`` first), so it raises."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RunIntegrityError(f"Corrupt answer journal {path}, line {number}: {error}")
    return records


class AnswerCheckpoint:
    """Durable per-question record of an answering run, so an interrupted run can resume.

    Every answer, including failed ones, is appended to a journal next to the answers file the
    moment it exists. With ``resume`` the journal is only appended to, never rewritten, so a
    crash at any point keeps every answer already in it. Records with status ``ok`` are reused
    and only the remaining questions are asked; failed and empty answers are asked again.
    Without ``resume`` the journal is restarted. Whether a journal may be reused at all is
    decided by the caller (the runner checks the manifest and config fingerprint).
    """

    def __init__(self, answers_path: str, questions: list[dict[str, Any]], resume: bool) -> None:
        self._path = journal_path(answers_path)
        self._questions = questions
        self._by_id: dict[str, list[dict[str, Any]]] = {}
        self.reused = 0

        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        if resume and os.path.exists(self._path):
            _drop_torn_tail(self._path)
            for record in _read_journal(self._path):
                if answer_status(record) is AnswerStatus.OK:
                    self._by_id.setdefault(record_id(record), []).append(record)
        elif not resume:
            open(self._path, "w", encoding="utf-8").close()

    def _append(self, record: dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def pending(self) -> list[dict[str, Any]]:
        """Questions that still need an answer, in their original order."""
        available = {qid: len(records) for qid, records in self._by_id.items()}
        pending = []
        for question in self._questions:
            qid = source_question_id(question)
            if available.get(qid, 0) > 0:
                available[qid] -= 1
                self.reused += 1
            else:
                pending.append(question)
        return pending

    def record(self, answer: dict[str, Any]) -> None:
        """Persist one finished answer. Passed to the answer executors as ``on_answer``."""
        self._append(answer)
        self._by_id.setdefault(record_id(answer), []).append(answer)

    def assemble(self) -> list[dict[str, Any]]:
        """All answers in question order: reused ones plus the ones recorded in this run."""
        queues = {qid: list(records) for qid, records in self._by_id.items()}
        answers = []
        for question in self._questions:
            qid = source_question_id(question)
            if not queues.get(qid):
                raise RunIntegrityError(f"No answer was recorded for question {question!r}")
            answers.append(queues[qid].pop(0))
        return answers
