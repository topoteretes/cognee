"""gliner_improve: the SLOW PASS that refines what the fast pass deferred.

The fast pass (`gliner_cognify`, default bank-selection schema) makes data
queryable in seconds with zero LLM calls, and records what it could not
judge: a `provisional` flag on the ontology, residue spans no selected
label covered, and — implicitly in the graph — chunks with few extracted
entities. This module spends LLM time on exactly that evidence:

  1. Load the dataset's cached ontology; skip if already refined and the
     residue is low (idempotent).
  2. Gather evidence: cached residue examples + the lowest-coverage chunk
     texts read back from the graph (fewest `contains` edges per character).
  3. ONE LLM call (`refine_schema`) proposes new entity types covering the
     residue and domain-specific relation types replacing the generic
     starter set. Additive only — existing graph nodes stay valid.
  4. Save the refined ontology (version bump, provisional=False) — every
     future fast-pass ingestion loads it from cache.
  5. Re-run the GLiNER pipeline over the dataset with the refined schema.
     Chunk ids are deterministic, so this upserts: existing nodes dedupe,
     new types add entities and relations. (A production version would
     re-extract only flagged chunks; the example re-runs the dataset —
     GLiNER is cheap, embeddings dedupe by id.)

Intended to run in the background or from cognee's improve() slot:

    await gliner_cognify(datasets=["docs"], workers=3)      # fast pass, seconds
    ...
    await gliner_improve("docs", extractor=extractor)       # slow pass, later
"""

import json
import pathlib
from collections import Counter

from cognee.shared.logging_utils import get_logger

from gliner_cognify import gliner_cognify
from ontology_schema import refine_schema

logger = get_logger("gliner_improve")

MAX_ENTITY_TYPES = 24  # measured recall ceiling — keep the schema lean


async def _low_coverage_chunk_texts(top_n: int = 5) -> list[str]:
    """Chunk texts with the fewest extracted entities per character.

    Reads the current graph context — call after a cognify run on the same
    dataset (as improve() normally is).
    """
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    chunks = {
        node_id: props
        for node_id, props in nodes
        if props.get("type") == "DocumentChunk" and props.get("text")
    }
    contains = Counter(source for source, _, relation, _ in edges if relation == "contains")
    ranked = sorted(
        chunks.items(),
        key=lambda item: contains.get(item[0], 0) / max(1, len(item[1]["text"])),
    )
    return [props["text"] for _, props in ranked[:top_n]]


async def gliner_improve(
    dataset: str,
    extractor=None,
    workers: int = 0,
    schema_cache_dir=None,
    residue_threshold: float = 0.05,
    max_new_types: int = 8,
    **cognify_kwargs,
):
    """Refine a provisional ontology with ONE LLM call and re-extract.

    Returns a report dict; no-op when the ontology is already refined and
    residue is below `residue_threshold`.
    """
    cache_dir = pathlib.Path(
        schema_cache_dir or pathlib.Path(__file__).parent / ".gliner_schema_cache"
    )
    cache_path = cache_dir / f"{dataset}.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached ontology for dataset {dataset!r} — run gliner_cognify first."
        )
    cached = json.loads(cache_path.read_text())
    residue = cached.get("residue", {})
    residue_ratio = residue.get("residue_ratio", 0.0)

    if not cached.get("provisional") and residue_ratio <= residue_threshold:
        logger.info(
            "Ontology for %r already refined (residue %.2f) — nothing to do.",
            dataset,
            residue_ratio,
        )
        return {"refined": False, "version": cached["version"]}

    entity_types = cached["entity_types"]
    relation_types = cached["relation_types"]
    samples = await _low_coverage_chunk_texts()

    logger.info(
        "Slow pass for %r: refining ontology v%d (residue %.2f, %d low-coverage samples) — ONE LLM call",
        dataset,
        cached["version"],
        residue_ratio,
        len(samples),
    )
    new_entities, new_relations = await refine_schema(
        entity_types,
        relation_types,
        residue.get("residue_examples", []),
        samples,
        max_new_types=max_new_types,
    )

    # Additive merge, capped at the measured label ceiling.
    room = max(0, MAX_ENTITY_TYPES - len(entity_types))
    added_entities = dict(list(new_entities.items())[:room])
    entity_types |= added_entities
    relation_types |= new_relations

    cached.update(
        {
            "version": cached["version"] + 1,
            "schema_source": "slow_pass_refined",
            "provisional": False,
            "entity_types": entity_types,
            "relation_types": relation_types,
            "refinement": {
                "added_entity_types": sorted(added_entities),
                "added_relation_types": sorted(new_relations),
            },
        }
    )
    cache_path.write_text(json.dumps(cached, indent=2))
    logger.info(
        "Ontology v%d saved: +%d entity types %s, +%d relation types %s",
        cached["version"],
        len(added_entities),
        sorted(added_entities),
        len(new_relations),
        sorted(new_relations),
    )

    # Re-extract with the refined schema (upsert: deterministic chunk ids).
    await gliner_cognify(
        datasets=[dataset],
        extractor=extractor,
        workers=workers,
        entity_types=entity_types,
        relation_types=relation_types,
        schema_cache_dir=cache_dir,
        **cognify_kwargs,
    )

    return {
        "refined": True,
        "version": cached["version"],
        "added_entity_types": sorted(added_entities),
        "added_relation_types": sorted(new_relations),
    }
