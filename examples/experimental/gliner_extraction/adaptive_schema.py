"""Adaptive schema expansion driven by per-batch entity density.

The schema-discovery LLM call in `ontology_schema.py` only sees the sample it
is given, so label sets can under-cover parts of a heterogeneous dataset.
This module closes that gap using GLiNER's own output as the coverage signal:

  1. After each extracted batch, compute entity density — entity mentions per
     1,000 characters of chunk text (spans are already available, so this is
     free).
  2. Keep a running average across batches. A batch whose density drops below
     `trigger_ratio` x the running average is evidence the current labels do
     not fit that region of the data.
  3. On trigger: sample the LOWEST-density chunks of that batch (the best
     evidence of what the schema misses), make ONE discovery LLM call, merge
     the proposed labels, and advise re-extracting the batch once with the
     expanded schema.

LLM cost is bounded by `max_discoveries` for the whole run; re-extraction
costs one extra GLiNER pass for the triggering batch only.
"""

import json
import pathlib

from cognee.shared.logging_utils import get_logger

from ontology_schema import discover_additional_types, discover_schema

logger = get_logger("adaptive_schema")


class AdaptiveSchemaTuner:
    """Tracks per-batch entity density and expands the schema on low coverage.

    Pass one instance for the whole run (it is stateful across batches):

        tuner = AdaptiveSchemaTuner(entity_types, relation_types)
        await gliner_cognify(..., entity_types=tuner.entity_types,
                             relation_types=tuner.relation_types,
                             schema_tuner=tuner)
    """

    def __init__(
        self,
        entity_types: dict,
        relation_types: dict,
        trigger_ratio: float = 0.6,
        min_batches: int = 2,
        max_discoveries: int = 5,
        sample_size: int = 5,
        cooldown_batches: int = 2,
    ):
        self.entity_types = dict(entity_types)
        self.relation_types = dict(relation_types)
        self.trigger_ratio = trigger_ratio
        self.min_batches = min_batches
        self.max_discoveries = max_discoveries
        self.sample_size = sample_size
        self.cooldown_batches = cooldown_batches

        self.batch_densities: list[float] = []
        self.discoveries: list[dict] = []
        self._cooldown = 0

    @staticmethod
    def _mentions(result: dict) -> int:
        return sum(len(values) for values in result.get("entities", {}).values())

    def _density(self, texts: list[str], results: list[dict]) -> float:
        chars = sum(len(t) for t in texts) or 1
        mentions = sum(self._mentions(r) for r in results)
        return mentions / chars * 1000

    async def observe_and_maybe_expand(self, texts: list[str], results: list[dict]) -> bool:
        """Record this batch's density; expand the schema if coverage looks low.

        Returns True when new labels were merged and the batch should be
        re-extracted with the expanded schema.
        """
        density = self._density(texts, results)
        history = self.batch_densities
        self.batch_densities.append(density)

        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if len(history) < self.min_batches or len(self.discoveries) >= self.max_discoveries:
            return False

        running_average = sum(history) / len(history)
        if density >= self.trigger_ratio * running_average:
            return False

        # Lowest-density chunks are the best evidence of what the schema misses.
        per_chunk = sorted(
            range(len(texts)),
            key=lambda i: self._mentions(results[i]) / (len(texts[i]) or 1),
        )[: self.sample_size]
        sample = [texts[i] for i in per_chunk]

        extra_entities, extra_relations = await discover_additional_types(
            sample, self.entity_types, self.relation_types
        )
        self._cooldown = self.cooldown_batches
        if not extra_entities and not extra_relations:
            return False

        self.entity_types.update(extra_entities)
        self.relation_types.update(extra_relations)
        self.discoveries.append(
            {
                "batch": len(self.batch_densities) - 1,
                "density": round(density, 3),
                "running_average": round(running_average, 3),
                "added_entity_types": sorted(extra_entities),
                "added_relation_types": sorted(extra_relations),
            }
        )
        logger.info(
            "Low entity density (%.2f vs avg %.2f) — schema expanded: +%d entity, +%d relation types",
            density,
            running_average,
            len(extra_entities),
            len(extra_relations),
        )
        return True

    def report(self) -> dict:
        """Post-run coverage summary."""
        densities = self.batch_densities
        return {
            "batches": len(densities),
            "mean_density_per_1k_chars": round(sum(densities) / len(densities), 3)
            if densities
            else 0.0,
            "min_density": round(min(densities), 3) if densities else 0.0,
            "max_density": round(max(densities), 3) if densities else 0.0,
            "discoveries": self.discoveries,
            "final_entity_types": sorted(self.entity_types),
            "final_relation_types": sorted(self.relation_types),
        }


class AutoSchemaManager(AdaptiveSchemaTuner):
    """Zero-configuration schema: cached per-dataset ontology + LLM discovery.

    The default schema source for `gliner_cognify`. Resolution order on the
    first batch of a run:

      1. Load the dataset's cached ontology (``<cache_dir>/<dataset>.json``,
         versioned) — no LLM call. This is the hot path for every ingestion
         after the first.
      2. No cache: ONE LLM ontology-discovery call over the first batch's
         texts (works with a small local model, e.g. Ollama qwen3:4b).
      3. LLM unavailable: fall back to the generic default labels, loudly.

    Density-triggered expansion (inherited from AdaptiveSchemaTuner) keeps
    working; every expansion bumps the cache version, so the ontology
    evolves across ingestions while the LLM stays out of the hot path.
    """

    def __init__(
        self,
        dataset: str,
        cache_dir: str | pathlib.Path,
        fallback_entity_types: dict | None = None,
        fallback_relation_types: dict | None = None,
        discovery_sample_size: int = 12,
        **tuner_kwargs,
    ):
        super().__init__({}, {}, **tuner_kwargs)
        self.dataset = dataset
        self.cache_path = pathlib.Path(cache_dir) / f"{dataset}.json"
        self.fallback_entity_types = fallback_entity_types
        self.fallback_relation_types = fallback_relation_types
        self.discovery_sample_size = discovery_sample_size
        self.version = 0
        self.schema_source = None
        self._initialized = False

    async def ensure_schema(self, texts: list[str]) -> None:
        """Resolve the schema once per run; called before the first extraction."""
        if self._initialized:
            return
        self._initialized = True

        if self.cache_path.exists():
            cached = json.loads(self.cache_path.read_text())
            self.entity_types = cached["entity_types"]
            self.relation_types = cached["relation_types"]
            self.version = cached["version"]
            self.schema_source = "cache"
            logger.info(
                "Ontology v%d loaded from cache for dataset %r (%d entity, %d relation types) — no LLM call",
                self.version,
                self.dataset,
                len(self.entity_types),
                len(self.relation_types),
            )
            return

        try:
            sample = texts[: self.discovery_sample_size]
            logger.info(
                "No cached ontology for dataset %r — running ONE LLM ontology-discovery call over %d sample chunks",
                self.dataset,
                len(sample),
            )
            self.entity_types, self.relation_types = await discover_schema(sample)
            self.schema_source = "llm_discovery"
        except Exception as error:
            from gliner_graph_extractor import DEFAULT_ENTITY_TYPES, DEFAULT_RELATION_TYPES

            self.entity_types = dict(self.fallback_entity_types or DEFAULT_ENTITY_TYPES)
            self.relation_types = dict(self.fallback_relation_types or DEFAULT_RELATION_TYPES)
            self.schema_source = "fallback_defaults"
            logger.warning(
                "Ontology discovery LLM call failed (%s: %s) — falling back to "
                "generic default labels. Configure an LLM (a local Ollama model "
                "works) or pass entity_types/relation_types explicitly.",
                type(error).__name__,
                error,
            )
        self._save()

    async def observe_and_maybe_expand(self, texts: list[str], results: list[dict]) -> bool:
        expanded = await super().observe_and_maybe_expand(texts, results)
        if expanded:
            self._save()
        return expanded

    def _save(self):
        self.version += 1
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "dataset": self.dataset,
                    "schema_source": self.schema_source,
                    "entity_types": self.entity_types,
                    "relation_types": self.relation_types,
                },
                indent=2,
            )
        )
        logger.info("Ontology v%d saved to %s", self.version, self.cache_path)
