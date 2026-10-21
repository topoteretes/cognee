"""Resolve relative time expressions in the windows before they are ingested (``--resolve-dates``).

Many LoCoMo questions ask when something happened, and their gold answers are phrased relative
to the session date ("the week before 9 June 2023") while the turns only say "last week". One
LLM call per window inserts, right after each relative expression, the date or period it refers
to, computed from the session date in the window header::

    Calvin: I'm heading there next month [= the month after 23 March 2023, i.e. April 2023].

The model may only add such annotations: an output that, with the annotations removed, is not
word for word the original window is rejected and the original window is kept. Annotations are
saved next to the memory (``date_annotations.json``) and reused when the same windows are ingested
again, so a re-run does not pay for them twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoConversation
from cognee.eval_framework.locomo.preprocess import SessionWindow
from cognee.shared.logging_utils import get_logger

logger = get_logger()

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "resolve_dates.txt"
SYSTEM_PROMPT = (
    "You annotate relative time expressions in an excerpt of a conversation. "
    "Reply with the excerpt only."
)
ANNOTATION = re.compile(r" ?\[= [^\]]*\]")
SESSION_DATE_FORMAT = "%I:%M %p on %d %B, %Y"  # "11:53 am on 23 March, 2023"
RETRIES = 3
DEFAULT_CONCURRENCY = 8


def session_datetime(date_time: str) -> datetime:
    # LoCoMo session dates carry no timezone; only the weekday and the date are used.
    return datetime.strptime(date_time.strip(), SESSION_DATE_FORMAT)  # noqa: DTZ007


def render_prompt(window_text: str, when: datetime, template: str | None = None) -> str:
    template = template if template is not None else PROMPT_PATH.read_text(encoding="utf-8")
    template = template.rstrip("\n")
    date = f"{when.day} {when.strftime('%B %Y')}"
    return (
        template.replace("{weekday}", when.strftime("%A"))
        .replace("{date}", date)
        .replace("{text}", window_text)
    )


def strip_annotations(text: str) -> str:
    return ANNOTATION.sub("", text)


def is_faithful(original: str, annotated: str) -> bool:
    """True when the annotated text is the original with annotations added and nothing else."""
    return strip_annotations(annotated).split() == original.split()


def window_key(window: SessionWindow) -> str:
    return f"{window.session_index}:{window.part}:{window.first_dia_id}"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def complete(prompt: str, *, model: str, api_key: str | None) -> str:
    """One annotation call. Low reasoning effort: the task is to copy text and add dates."""
    import litellm

    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        reasoning_effort="low",
        drop_params=True,  # ignores reasoning_effort on models that do not support it
    )
    return response.choices[0].message.content or ""


async def annotate_text(
    text: str,
    when: datetime,
    *,
    model: str,
    api_key: str | None,
    semaphore: asyncio.Semaphore,
    template: str | None = None,
) -> tuple[str, str]:
    """Return ``(text, status)``: the annotated text, or the original with a reason."""
    prompt = render_prompt(text, when, template)
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                output = (await complete(prompt, model=model, api_key=api_key)).strip("\n")
                break
            except Exception as error:  # transient provider errors: retry, then keep the window
                if attempt == RETRIES - 1:
                    logger.exception("Date annotation failed; keeping the window as it is")
                    return text, f"error: {type(error).__name__}: {error}"
                await asyncio.sleep(2.0 * (attempt + 1))
    if not is_faithful(text, output):
        return text, "rejected: text changed"
    return output, "annotated" if output != text else "unchanged"


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8")).get("windows", [])
    return {row["key"]: row for row in rows}


async def resolve_window_dates(
    conversation: LocomoConversation,
    windows: list[SessionWindow],
    *,
    model: str,
    api_key: str | None,
    cache_path: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[list[SessionWindow], dict[str, Any]]:
    """Annotate every window (reusing ``cache_path`` where the window text is unchanged)."""
    session_dates = {
        session.index: session_datetime(session.date_time) for session in conversation.sessions
    }
    template = PROMPT_PATH.read_text(encoding="utf-8")
    cached = load_annotations(cache_path)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(window: SessionWindow) -> dict[str, Any]:
        key = window_key(window)
        source_hash = _text_hash(window.text)
        row = cached.get(key)
        # Reuse earlier results for the same text, so the ingested text is identical on a re-run;
        # only windows whose call failed are tried again.
        if (
            row
            and row.get("source_sha256") == source_hash
            and not row["status"].startswith("error")
        ):
            return {**row, "reused": True}
        text, status = await annotate_text(
            window.text,
            session_dates[window.session_index],
            model=model,
            api_key=api_key,
            semaphore=semaphore,
            template=template,
        )
        return {
            "key": key,
            "session_index": window.session_index,
            "part": window.part,
            "first_dia_id": window.first_dia_id,
            "source_sha256": source_hash,
            "status": status,
            "annotations": len(ANNOTATION.findall(text)),
            "text": text,
            "reused": False,
        }

    rows = await asyncio.gather(*(one(window) for window in windows))
    stats = {
        "model": model,
        "windows": len(rows),
        "reused": sum(1 for row in rows if row["reused"]),
        "annotations": sum(row["annotations"] for row in rows),
    }
    for status in ("annotated", "unchanged", "rejected", "error"):
        stats[status] = sum(1 for row in rows if row["status"].split(":")[0] == status)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "conversation": conversation.sample_id,
                "model": model,
                "prompt": PROMPT_PATH.name,
                "windows": [{k: v for k, v in row.items() if k != "reused"} for row in rows],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved = [
        replace(window, text=row["text"], word_count=len(row["text"].split()))
        for window, row in zip(windows, rows)
    ]
    return resolved, stats
