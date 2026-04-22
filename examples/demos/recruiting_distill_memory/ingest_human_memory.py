"""Ingest the Ledgerline human memory: rulebook + prose playbook.

Builds the `human_memory` dataset the agent will retrieve from at runtime.

  1. Prose playbook  (alex_playbook.md) → cognee.add + cognify.
     Chunked, embedded, summarized — gives semantic rationale context.
  2. Structured rules (seed_rules.yaml) → Rule DataPoints persisted via a
     one-task custom pipeline so they're scoped to `human_memory` with
     user/dataset/ACL wiring. Tagged belongs_to_set=['rule','approved'].

Run:
    python -m examples.demos.recruiting_distill_memory.ingest_human_memory
    python -m examples.demos.recruiting_distill_memory.ingest_human_memory --reset
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml

import cognee
from cognee.infrastructure.databases.relational.create_db_and_tables import (
    create_db_and_tables,
)
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.tasks.storage import add_data_points

from examples.demos.recruiting_distill_memory.rule_models import Rule


HERE = Path(__file__).parent
PLAYBOOK = HERE / "data" / "alex_playbook.md"
SEED_RULES = HERE / "data" / "seed_rules.yaml"
DATASET = "human_memory"


def load_seed_rules() -> list[Rule]:
    raw = yaml.safe_load(SEED_RULES.read_text())
    rules: list[Rule] = []
    for record in raw["rules"]:
        rule = Rule.model_validate(record)
        # Tag as human_authored so agent-proposed rules (tagged
        # 'agent_authored' by review_pending_rules.py) stay visually and
        # programmatically distinct after both have status='approved'.
        rule.belongs_to_set = ["rule", rule.status, "human_authored"]
        rules.append(rule)
    return rules


async def _persist_rules_task(_data: Any, ctx: PipelineContext = None) -> list[Rule]:
    rules = load_seed_rules()
    await add_data_points(rules, ctx=ctx)
    return rules


async def main(reset: bool) -> None:
    if reset:
        print("Pruning cognee data + system metadata...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    # Prune wipes the SQLite metadata tables; recreate them before ingesting.
    await create_db_and_tables()

    print(f"Ingesting prose playbook → dataset '{DATASET}' (node_set=['rationale'])")
    await cognee.add(
        str(PLAYBOOK),
        dataset_name=DATASET,
        node_set=["rationale"],
    )

    print(f"Running cognify on '{DATASET}' (chunks, embeddings, summaries)...")
    await cognee.cognify(datasets=[DATASET])

    print(f"Persisting structured rules → dataset '{DATASET}' (as typed Rule nodes)")
    user = await get_default_user()
    await cognee.run_custom_pipeline(
        tasks=[Task(_persist_rules_task)],
        data=[None],
        dataset=DATASET,
        user=user,
        pipeline_name="ingest_seed_rules",
    )

    rules = load_seed_rules()
    print(f"\nDone. human_memory now contains:")
    print(f"  - prose rationale from {PLAYBOOK.name}")
    print(f"  - {len(rules)} Rule DataPoints:")
    for rule in rules:
        print(f"      • {rule.rule_id} [{rule.domain}/{rule.status}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Prune cognee data + system metadata before ingesting.",
    )
    args = parser.parse_args()
    asyncio.run(main(reset=args.reset))
