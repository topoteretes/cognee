"""Compare DEFAULT / AUTO_RESTRICTED / AUTO_RESTRICTED_ITERATIVE ontology generation.

For each approach: prune state, set ONTOLOGY_GENERATION, ingest the 5 CVs from
simple_cognee_example, cognify, then collect the entity types (EntityType
node names) and edge types (extracted relationship_name values, excluding
pipeline scaffolding) from the resulting graph. Renders the per-approach
results into a self-contained HTML dashboard for side-by-side comparison.
"""

import asyncio
import html
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.logging_utils import ERROR, setup_logging

from simple_cognee_example import job_1, job_2, job_3, job_4, job_5

JOBS = [job_1, job_2, job_3, job_4, job_5]

# Pipeline scaffolding relations — not part of LLM-extracted edge types.
SYSTEM_EDGE_TYPES = {"is_a", "contains", "made_from"}

RUNS = [
    ("Default", "DEFAULT"),
    ("AUTO_RESTRICTED", "AUTO_RESTRICTED"),
    ("AUTO_RESTRICTED_ITERATIVE", "AUTO_RESTRICTED_ITERATIVE"),
]


async def collect_types() -> tuple[set[str], set[str]]:
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    entity_types: set[str] = set()
    for _node_id, props in nodes:
        props = props or {}
        if props.get("type") == "EntityType":
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                entity_types.add(name.strip())

    edge_types: set[str] = set()
    for edge in edges:
        if len(edge) < 3:
            continue
        rel = edge[2]
        if isinstance(rel, str) and rel.strip() and rel not in SYSTEM_EDGE_TYPES:
            edge_types.add(rel.strip())

    return entity_types, edge_types


async def run_approach(name: str, ontology_generation: str) -> dict:
    print(f"\n=== {name} (ONTOLOGY_GENERATION={ontology_generation}) ===")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Pruned.")

    get_cognify_config().ontology_generation = ontology_generation

    for text in JOBS:
        await cognee.add(text)
    print(f"Added {len(JOBS)} CVs.")

    await cognee.cognify()
    print("Cognify done.")

    entity_types, edge_types = await collect_types()
    print(f"  entity types: {len(entity_types)}, edge types: {len(edge_types)}")

    return {
        "name": name,
        "ontology_generation": ontology_generation,
        "entity_types": sorted(entity_types),
        "edge_types": sorted(edge_types),
    }


def _comparison_table(category: str, all_items: list[str], results: list[dict]) -> str:
    key = "entity_types" if category == "entity" else "edge_types"
    header_cells = "<th>Type</th>" + "".join(
        f"<th>{html.escape(r['name'])}</th>" for r in results
    )
    rows = []
    for item in all_items:
        cells = [f"<td class='label'>{html.escape(item)}</td>"]
        for r in results:
            present = item in r[key]
            cells.append(
                f"<td class='cell {'present' if present else 'absent'}'>"
                f"{'&check;' if present else '&mdash;'}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_dashboard(results: list[dict], output_path: Path) -> None:
    all_entity_types = sorted({et for r in results for et in r["entity_types"]})
    all_edge_types = sorted({et for r in results for et in r["edge_types"]})

    cards = []
    for r in results:
        et_items = (
            "".join(f"<li>{html.escape(t)}</li>" for t in r["entity_types"])
            or "<li><em>none</em></li>"
        )
        edt_items = (
            "".join(f"<li>{html.escape(t)}</li>" for t in r["edge_types"])
            or "<li><em>none</em></li>"
        )
        cards.append(
            f"""
            <div class="card">
              <h3>{html.escape(r["name"])}</h3>
              <p class="config">ONTOLOGY_GENERATION = <code>{html.escape(r["ontology_generation"])}</code></p>
              <div class="counts">
                <span class="pill">{len(r["entity_types"])} entity types</span>
                <span class="pill">{len(r["edge_types"])} edge types</span>
              </div>
              <h4>Entity types</h4>
              <ul>{et_items}</ul>
              <h4>Edge types</h4>
              <ul>{edt_items}</ul>
            </div>
            """
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Auto-Restricted Ontology Comparison</title>
<style>
  :root {{
    --border: #e5e7eb; --muted: #6b7280; --text: #111827;
    --present-bg: #ecfdf5; --present-fg: #15803d; --absent-fg: #cbd5e1;
    --pill-bg: #eef2ff; --pill-fg: #3730a3;
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 2rem auto; max-width: 1180px; padding: 0 1.5rem;
    color: var(--text); line-height: 1.45; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 .25rem; }}
  .subtitle {{ color: var(--muted); margin: 0 0 2rem; }}
  h2 {{ margin: 2.5rem 0 1rem; font-size: 1.15rem;
    border-bottom: 1px solid var(--border); padding-bottom: .35rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
  .card {{ border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.1rem;
    background: #fafafa; display: flex; flex-direction: column; }}
  .card h3 {{ margin: 0; font-size: 1.05rem; }}
  .card h4 {{ font-size: .85rem; margin: .9rem 0 .3rem; color: #374151;
    text-transform: uppercase; letter-spacing: .03em; }}
  .config {{ color: var(--muted); font-size: .82rem; margin: .15rem 0 .6rem; }}
  .counts {{ display: flex; gap: .35rem; margin: .25rem 0 .5rem; }}
  .pill {{ background: var(--pill-bg); color: var(--pill-fg);
    padding: 2px 8px; border-radius: 999px; font-size: .75rem; font-weight: 600; }}
  ul {{ margin: 0; padding-left: 1.2rem; font-size: .85rem;
    max-height: 220px; overflow-y: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem;
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: .45rem .7rem; text-align: left; border-bottom: 1px solid #f1f5f9; }}
  th {{ background: #f8fafc; font-weight: 600; font-size: .85rem; }}
  td.label {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; }}
  td.cell {{ text-align: center; width: 8rem; font-family: monospace; }}
  td.present {{ color: var(--present-fg); background: var(--present-bg); font-weight: 700; }}
  td.absent {{ color: var(--absent-fg); }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: .85em; }}
</style>
</head>
<body>
<h1>Auto-Restricted Ontology &mdash; Approach Comparison</h1>
<p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &middot; 5 CVs ingested via simple_cognee_example</p>

<h2>Per-approach results</h2>
<div class="cards">{"".join(cards)}</div>

<h2>Entity types &mdash; coverage across approaches ({len(all_entity_types)} unique)</h2>
{_comparison_table("entity", all_entity_types, results)}

<h2>Edge types &mdash; coverage across approaches ({len(all_edge_types)} unique)</h2>
{_comparison_table("edge", all_edge_types, results)}
</body>
</html>
"""
    output_path.write_text(html_doc)


async def main():
    results = []
    for name, ontology_generation in RUNS:
        results.append(await run_approach(name, ontology_generation))

    output_path = SCRIPT_DIR / "auto_restricted_ontology_comparison.html"
    render_dashboard(results, output_path)
    print(f"\nDashboard: {output_path}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
