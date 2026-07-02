"""Citation extraction for cognee-powered bots.

``cognee.recall(..., include_references=True)`` returns a list of result
entries (session QA hits, graph completions, graph context, ...). Agents
and end users both need to *attribute* an answer to its sources, so this
module normalizes those heterogeneous entries into a flat list of
``Citation`` objects the transport layer can render inline.

The extractor is intentionally defensive: results may arrive as Pydantic
models (``RecallResponse``) or plain dicts, and references may live under
``metadata``/``raw`` depending on the retriever. When no explicit
reference is present we fall back to a snippet of the entry text so a
reply is never uncited.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, List, Optional

# Score in cognee is a raw backend distance where *lower is better*. At the
# surface we want the intuitive direction (higher = more relevant), matching
# the ranking work in issue #3604.
_MAX_SNIPPET = 240


@dataclasses.dataclass
class Citation:
    """One attributable source behind an answer."""

    source: str  # "session" | "graph" | "graph_context" | "docs" | ...
    snippet: str
    score: Optional[float] = None
    dataset: Optional[str] = None
    reference: Optional[str] = None  # data_id / chunk_id / url when available

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _to_dict(entry: Any) -> dict:
    """Normalize a Pydantic model or dict result entry into a plain dict."""
    if isinstance(entry, dict):
        return entry
    for attr in ("model_dump", "dict"):
        fn = getattr(entry, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - best-effort normalization
                pass
    return {k: v for k, v in vars(entry).items() if not k.startswith("_")}


def _snippet(text: Optional[str]) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX_SNIPPET else text[: _MAX_SNIPPET - 1] + "…"


def _reference_id(ref: dict) -> Optional[str]:
    for key in ("url", "reference", "chunk_id", "data_id", "id"):
        value = ref.get(key)
        if value:
            return str(value)
    return None


def _references_from(entry: dict) -> List[dict]:
    """Pull an explicit references list from wherever the retriever put it."""
    for container in (entry, entry.get("metadata") or {}, entry.get("raw") or {}):
        refs = container.get("references") if isinstance(container, dict) else None
        if isinstance(refs, list) and refs:
            return [r for r in refs if isinstance(r, dict)]
    return []


def extract_citations(results: Iterable[Any]) -> List[Citation]:
    """Turn ``recall`` results into a flat, de-duplicated citation list."""
    citations: List[Citation] = []
    seen: set = set()

    for entry in results or []:
        data = _to_dict(entry)
        source = str(data.get("source") or "graph")
        dataset = data.get("dataset_name") or data.get("dataset")
        score = data.get("score")

        explicit = _references_from(data)
        if explicit:
            for ref in explicit:
                snippet = _snippet(ref.get("snippet") or ref.get("text"))
                key = (snippet, _reference_id(ref))
                if not snippet or key in seen:
                    continue
                seen.add(key)
                citations.append(
                    Citation(
                        source=source,
                        snippet=snippet,
                        score=ref.get("score", score),
                        dataset=ref.get("dataset") or dataset,
                        reference=_reference_id(ref),
                    )
                )
            continue

        # Fallback: cite the entry itself so a reply is never uncited.
        snippet = _snippet(data.get("text") or data.get("answer") or data.get("content"))
        key = (snippet, None)
        if snippet and key not in seen:
            seen.add(key)
            citations.append(Citation(source=source, snippet=snippet, score=score, dataset=dataset))

    return citations
