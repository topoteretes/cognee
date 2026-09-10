"""Closed-schema resolution for the GLiNER extraction path.

Each document receives a closed schema through this fallback chain — first
non-empty source wins:

1. labels the caller passed to ``get_gliner_tasks``,
2. else classes / object properties of the configured OWL ontology,
3. else the frozen label banks, filtered to the labels that fire on a document
   sketch.

Explicit caller labels short-circuit the chain: no ontology file is read and no
probing pass runs over the chunks.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from cognee.shared.logging_utils import get_logger

from .banks import LABEL_BANK, RELATION_BANK

logger = get_logger("gliner.schema")

# Upper bound on the number of entity types (and, separately, relation types)
# sent to GLiNER in one schema. Label quality drops as the closed set grows.
MAX_TYPES = 20
MAX_SKETCH_CHARS = 12_000
MAX_SKETCH_WORDS = 3_000

LabelSpec = Sequence[str] | Mapping[str, str | None]


@dataclass(frozen=True)
class GlinerSchema:
    """Entity and relation type names with optional descriptions.

    Both mappings are ``name -> description``; the description is ``""`` when
    none is known. ``source`` records which step of the fallback chain produced
    the schema (``caller`` / ``ontology`` / ``label_bank`` / ``empty``).
    """

    entity_types: dict[str, str] = field(default_factory=dict)
    relation_types: dict[str, str] = field(default_factory=dict)
    source: str = "empty"

    @property
    def is_empty(self) -> bool:
        return not self.entity_types and not self.relation_types


EMPTY_SCHEMA = GlinerSchema()


def as_label_map(spec: LabelSpec | None) -> dict[str, str]:
    """Normalize a list of names or a ``name -> description`` mapping to a dict."""
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        items = ((str(name), str(description or "")) for name, description in spec.items())
    elif isinstance(spec, str):
        items = ((spec, ""),)
    else:
        items = ((str(name), "") for name in spec)
    result: dict[str, str] = {}
    for name, description in items:
        name = name.strip()
        if name and name not in result:
            result[name] = description.strip()
    return result


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^0-9a-zA-Z]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_MONEY_RE = re.compile(r"(?:[$€£]\s?\d|(?:USD|EUR|GBP)\s?\d|\d+(?:\.\d+)?%)", re.IGNORECASE)
_CONTACT_RE = re.compile(r"(?:\b[\w.+-]+@[\w.-]+\.\w+\b|https?://\S+)")
_ID_RE = re.compile(r"\b[A-Z]{2,}[-_/]?\d{2,}\b")
_CAPS_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def to_snake_case(name: str) -> str:
    """``Person`` -> ``person``, ``worksAt`` -> ``works_at``, ``HTTP Server`` -> ``http_server``."""
    name = _CAMEL_BOUNDARY.sub("_", str(name).strip())
    name = _NON_WORD.sub("_", name)
    return re.sub(r"_+", "_", name).strip("_").lower()


def make_document_sketch(
    text: str,
    max_chars: int = MAX_SKETCH_CHARS,
    max_words: int = MAX_SKETCH_WORDS,
) -> str:
    """Keep a small mix of high-signal and evenly distributed sentences."""

    def limit_words(value: str) -> str:
        words = list(re.finditer(r"\S+", value))
        return value if len(words) <= max_words else value[: words[max_words - 1].end()]

    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(text) if sentence.strip()]
    joined = "\n".join(sentences)
    if len(joined) <= max_chars:
        return limit_words(joined)

    def score(sentence: str) -> int:
        value = 3 * bool(_DATE_RE.search(sentence))
        value += 2 * bool(_NUMBER_RE.search(sentence))
        value += 3 * bool(_MONEY_RE.search(sentence))
        value += 3 * bool(_CONTACT_RE.search(sentence))
        value += 3 * bool(_ID_RE.search(sentence))
        value += min(3, len(_CAPS_RE.findall(sentence)))
        if len(sentence) < 100 and (
            sentence.isupper() or sentence.endswith(":") or sentence.istitle()
        ):
            value += 4
        return value

    scores = [score(sentence) for sentence in sentences]
    signal_budget = int(max_chars * 0.75)
    chosen: dict[int, str] = {}
    used = 0
    ranked = sorted(
        (index for index, value in enumerate(scores) if value),
        key=lambda index: scores[index],
        reverse=True,
    )
    for index in ranked:
        sentence = sentences[index]
        cost = len(sentence) + bool(chosen)
        if used + cost <= signal_budget:
            chosen[index] = sentence
            used += cost

    remaining = [index for index in range(len(sentences)) if index not in chosen]
    available = max_chars - used - bool(chosen)
    average_length = max(1, sum(len(sentences[index]) for index in remaining) // len(remaining))
    sample_count = min(len(remaining), max(1, available // average_length))
    for sample in range(sample_count):
        position = round(sample * (len(remaining) - 1) / max(1, sample_count - 1))
        index = remaining[position]
        separator = bool(chosen)
        available = max_chars - used - separator
        if available <= 0:
            break
        share = max(1, available // (sample_count - sample))
        sentence = sentences[index][:share].rsplit(" ", 1)[0] or sentences[index][:share]
        chosen[index] = sentence
        used += len(sentence) + separator

    return limit_words("\n".join(chosen[index] for index in sorted(chosen)))


def cap_types(types: Mapping[str, str], hit_counts: Mapping[str, int] | None = None) -> dict:
    """Keep at most ``MAX_TYPES`` labels: by hit count (desc) when given, then by name."""
    if hit_counts is None:
        ordered = sorted(types)
    else:
        ordered = sorted(types, key=lambda name: (-hit_counts.get(name, 0), name))
    return {name: types[name] for name in ordered[:MAX_TYPES]}


# --------------------------------------------------------------------------- #
# Step 2: ontology
# --------------------------------------------------------------------------- #


def _local_name(uri: str) -> str:
    uri = str(uri).rstrip("/#")
    for separator in ("#", "/", ":"):
        if separator in uri:
            uri = uri.rsplit(separator, 1)[-1]
    return uri


def _collect_ontology_terms(graph: Any, rdf_type: Any) -> dict[str, str]:
    from rdflib import RDF, RDFS, URIRef

    collected: dict[str, str] = {}
    for subject in sorted(set(graph.subjects(RDF.type, rdf_type)), key=str):
        if not isinstance(subject, URIRef):
            continue
        label = graph.value(subject, RDFS.label)
        name = to_snake_case(str(label) if label is not None else _local_name(subject))
        if not name:
            continue
        comment = graph.value(subject, RDFS.comment)
        description = str(comment).strip() if comment is not None else ""
        if name not in collected or (description and not collected[name]):
            collected[name] = description
    return collected


def schema_from_ontology(ontology_file_path: str | None = None) -> GlinerSchema:
    """Derive entity and relation type names from an OWL ontology.

    OWL classes become entity types and OWL object properties become relation
    types. Names are the snake_case of ``rdfs:label`` when present, else of the
    local name (``Person`` -> ``person``, ``worksAt`` -> ``works_at``);
    ``rdfs:comment`` is used as the label description. With no configured file,
    an unreadable file, or nothing mapped, the result is empty.
    """
    path = ontology_file_path
    if path is None:
        from cognee.modules.ontology.ontology_env_config import get_ontology_env_config

        path = get_ontology_env_config().ontology_file_path
    if not path or not os.path.isfile(path):
        return EMPTY_SCHEMA

    from rdflib import OWL, Graph

    graph = Graph()
    parsed = False
    for fmt in (None, "xml", "turtle"):
        try:
            graph.parse(path, format=fmt)
            parsed = True
            break
        except Exception as error:  # noqa: BLE001 - try the next format
            logger.debug("Ontology parse with format=%s failed: %s", fmt, error)
    if not parsed:
        logger.warning("Could not parse ontology file %s; GLiNER schema falls through", path)
        return EMPTY_SCHEMA

    entity_types = _collect_ontology_terms(graph, OWL.Class)
    relation_types = _collect_ontology_terms(graph, OWL.ObjectProperty)
    if not entity_types and not relation_types:
        return EMPTY_SCHEMA
    return GlinerSchema(cap_types(entity_types), cap_types(relation_types), source="ontology")


# --------------------------------------------------------------------------- #
# Step 3: label bank probe
# --------------------------------------------------------------------------- #


def _count_hits(results: Sequence[Mapping[str, Any]], section: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for name, hits in (result or {}).get(section, {}).items():
            counts[name] = counts.get(name, 0) + len(hits or [])
    return counts


def schema_from_label_bank(
    extractor: Any,
    sketch: str,
    *,
    threshold: float,
) -> GlinerSchema:
    """Probe one document sketch and keep only bank labels that fired.

    Returned names are always members of ``LABEL_BANK`` / ``RELATION_BANK``,
    ordered by hit count then name and capped at ``MAX_TYPES`` each. Empty when
    nothing fired (or the sketch is empty).
    """
    from .extractor import extract_once

    if not sketch:
        return EMPTY_SCHEMA

    probe_schema = GlinerSchema(dict(LABEL_BANK), dict(RELATION_BANK), source="label_bank")
    results = [extract_once(extractor, sketch, probe_schema, threshold=threshold)]

    entity_hits = {
        n: c for n, c in _count_hits(results, "entities").items() if c and n in LABEL_BANK
    }
    relation_hits = {
        n: c
        for n, c in _count_hits(results, "relation_extraction").items()
        if c and n in RELATION_BANK
    }
    if not entity_hits and not relation_hits:
        return EMPTY_SCHEMA

    entity_types = cap_types({n: LABEL_BANK[n] for n in entity_hits}, entity_hits)
    relation_types = cap_types({n: RELATION_BANK[n] for n in relation_hits}, relation_hits)
    return GlinerSchema(entity_types, relation_types, source="label_bank")


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #


def _validate_caller_labels(kind: str, labels: Mapping[str, str]) -> None:
    if len(labels) > MAX_TYPES:
        raise ValueError(
            f"Cognee supports at most {MAX_TYPES} GLiNER {kind}; {len(labels)} were given. "
            "Trim the list rather than relying on a silent cut."
        )


def resolve_schema(
    entity_types: LabelSpec | None = None,
    relation_types: LabelSpec | None = None,
    *,
    extractor: Any = None,
    probe_text: str = "",
    ontology_file_path: str | None = None,
    threshold: float = 0.5,
) -> GlinerSchema:
    """Resolve caller labels, an ontology, or one document sketch in that order.

    Caller labels win outright (steps 2 and 3 do not run). Otherwise the
    configured ontology is tried, then the label banks are probed on
    ``probe_text`` — which requires ``extractor``.
    """
    caller_entities = as_label_map(entity_types)
    caller_relations = as_label_map(relation_types)
    if caller_entities or caller_relations:
        _validate_caller_labels("entity types", caller_entities)
        _validate_caller_labels("relation types", caller_relations)
        return GlinerSchema(caller_entities, caller_relations, source="caller")

    from_ontology = schema_from_ontology(ontology_file_path)
    if not from_ontology.is_empty:
        return from_ontology

    if extractor is None:
        return EMPTY_SCHEMA
    return schema_from_label_bank(
        extractor,
        probe_text,
        threshold=threshold,
    )
