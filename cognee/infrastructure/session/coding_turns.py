"""The turns of a coding session and which traces each one owns (SDK-1335).

A turn is one user request, the traces the agent produced for it and the agent's final
answer: the unit of work the coding distillation learns from. The session cache keeps
the requests and answers as Q&A rows and the traces as a separate list, so the two have
to be paired here. Positions cannot do it: the Claude Code plugin writes the answer from
an asynchronous Stop hook, so the first traces of the next turn can land before it, and
a subagent left running in the background keeps writing traces after the turn is over.
Times can: every entry carries the moment of its event, stamped by the client when it
sends one (``QAEntry.time``, ``TraceEntry.time``) and by the store otherwise.

The rule, one per turn:

- A turn starts when its prompt was submitted (the Q&A row's ``prompt_time``, which the
  plugin stamps), else when the previous turn's answer was stored; the first turn
  without a prompt time has no start. It ends at its answer's time.
- A trace belongs to the latest turn that had started when the trace happened, else to
  the first turn. A trace between one turn's answer and the next turn's prompt is the
  late work of the earlier turn. A trace with no time (a row written before the field
  existed) belongs to the newest turn, as every trace did before.
- A trace newer than the last answer has no answer yet: it and everything stored after
  it wait for the next improve (``settled_traces``), so the extraction watermark stays a
  plain count of the trace list.
"""

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class Turn:
    """One Q&A row of the session with the traces it owns, oldest first."""

    qa_id: str
    request: str
    final_answer: str
    qa_context: dict
    start: datetime | None
    end: datetime | None
    traces: list = field(default_factory=list)


def parse_qa_context(context: str | None) -> dict:
    """The structured context a coding plugin stores on its Q&A entry, or ``{}``.

    SDK-1336 sends ``{"cwd", "transcript_path", "date", "start_commit", "end_commit",
    "branch"}`` as the entry's ``context``; any other text is not an error.
    """
    if not context or not isinstance(context, str):
        return {}
    try:
        parsed = json.loads(context)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_event_time(value) -> datetime | None:
    """An entry's ISO-8601 time as an aware datetime; None when absent or unparsable.
    A naive value is read as UTC, which is what the stores stamp."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _field(row, name: str):
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def build_turns(qa_rows: list) -> list[Turn]:
    """The session's Q&A rows as turns in time order, without traces."""
    rows = sorted(
        qa_rows,
        key=lambda row: (
            parse_event_time(_field(row, "time")) is None,
            parse_event_time(_field(row, "time")) or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )
    turns: list[Turn] = []
    previous_end: datetime | None = None
    for row in rows:
        qa_context = parse_qa_context(_field(row, "context"))
        end = parse_event_time(_field(row, "time"))
        turns.append(
            Turn(
                qa_id=str(_field(row, "qa_id") or ""),
                request=str(_field(row, "question") or ""),
                final_answer=str(_field(row, "answer") or ""),
                qa_context=qa_context,
                start=parse_event_time(_field(row, "prompt_time")) or previous_end,
                end=end,
            )
        )
        previous_end = end
    return turns


def owning_turn_index(turns: list[Turn], trace) -> int:
    """The index of the turn a trace belongs to (see the module docstring)."""
    if not turns:
        raise ValueError("a trace cannot be assigned without turns")
    happened = parse_event_time(getattr(trace, "time", None))
    if happened is None:
        return len(turns) - 1
    owner = 0
    for index, turn in enumerate(turns):
        if turn.start is not None and turn.start <= happened:
            owner = index
    return owner


def settled_traces(traces: list, turns: list[Turn]) -> list:
    """The leading traces whose answer exists: everything before the first trace newer
    than the last turn's answer. Without a turn nothing is settled."""
    if not turns:
        return []
    last_end = turns[-1].end
    settled = []
    for trace in traces:
        happened = parse_event_time(getattr(trace, "time", None))
        if happened is not None and last_end is not None and happened > last_end:
            break
        settled.append(trace)
    return settled


def group_by_turn(traces: list, turns: list[Turn]) -> list[Turn]:
    """The turns that own ``traces``, oldest first, each with its traces in the given order."""
    owned: dict[int, list] = {}
    for trace in traces:
        owned.setdefault(owning_turn_index(turns, trace), []).append(trace)
    return [replace(turns[index], traces=owned[index]) for index in sorted(owned)]
