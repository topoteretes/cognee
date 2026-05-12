"""Compare DEFAULT / AUTO_RESTRICTED / AUTO_RESTRICTED_ITERATIVE ontology generation.

For each approach: prune state, set ONTOLOGY_GENERATION, ingest the 5 CVs from
simple_cognee_example, cognify, then collect the entity types (EntityType
node names) and edge types (extracted relationship_name values, excluding
pipeline scaffolding) from the resulting graph. Renders the per-approach
results into a self-contained HTML dashboard for side-by-side comparison.
"""

import asyncio
import html
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import litellm
from litellm.integrations.custom_logger import CustomLogger

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.logging_utils import ERROR, setup_logging

from simple_cognee_example import job_1, job_2, job_3, job_4, job_5

JOBS = [job_1, job_2, job_3, job_4, job_5]

# How many sample triplets to surface per approach in the dashboard.
TRIPLET_SAMPLE_SIZE = 6
TRIPLET_SAMPLE_SEED = 7  # deterministic samples across runs

RUNS = [
    ("Default", "DEFAULT"),
    ("AUTO_RESTRICTED", "AUTO_RESTRICTED"),
    ("AUTO_RESTRICTED_ITERATIVE", "AUTO_RESTRICTED_ITERATIVE"),
]


@dataclass
class UsageTracker:
    """Per-approach LLM telemetry."""

    # `calls` is counted at the LLMGateway boundary so it's robust even if
    # litellm's callback hooks miss some calls. `prompt_tokens` /
    # `completion_tokens` come from litellm's CustomLogger which sees the
    # raw response with usage info that LLMGateway strips out before returning.
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


TRACKER = UsageTracker()


def _extract_tokens(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None) or (
        response.get("usage") if isinstance(response, dict) else None
    )
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


class _UsageLogger(CustomLogger):
    """Captures token usage from every successful litellm completion."""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        prompt, completion = _extract_tokens(response_obj)
        TRACKER.prompt_tokens += prompt
        TRACKER.completion_tokens += completion

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        prompt, completion = _extract_tokens(response_obj)
        TRACKER.prompt_tokens += prompt
        TRACKER.completion_tokens += completion


# `litellm.callbacks` is the modern slot that accepts CustomLogger instances
# for both sync and async paths. `success_callback` only accepts string IDs
# of built-in integrations in current litellm versions.
litellm.callbacks = [_UsageLogger()]


# Monkey-patch the gateway so call count is captured regardless of how
# litellm callbacks behave. Demo-only, runtime patch, never persisted.
_original_acreate = LLMGateway.acreate_structured_output


def _tracked_acreate(text_input, system_prompt, response_model, **kwargs):
    TRACKER.calls += 1
    return _original_acreate(text_input, system_prompt, response_model, **kwargs)


LLMGateway.acreate_structured_output = staticmethod(_tracked_acreate)


async def collect_stats() -> dict:
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    entity_types: set[str] = set()
    node_types: set[str] = set()
    entity_count = 0
    id_to_name: dict[str, str] = {}
    entity_ids: set[str] = set()
    for node_id, props in nodes:
        props = props or {}
        node_type = props.get("type")
        if isinstance(node_type, str) and node_type:
            node_types.add(node_type)
        name = props.get("name")
        if isinstance(name, str) and name.strip():
            id_to_name[str(node_id)] = name.strip()
        if node_type == "Entity":
            entity_count += 1
            entity_ids.add(str(node_id))
        if node_type == "EntityType":
            if isinstance(name, str) and name.strip():
                entity_types.add(name.strip())

    edge_types: set[str] = set()
    triplet_pool: list[tuple[str, str, str]] = []
    for edge in edges:
        if len(edge) < 3:
            continue
        src, tgt, rel = str(edge[0]), str(edge[1]), edge[2]
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel = rel.strip()
        edge_types.add(rel)
        if src in entity_ids and tgt in entity_ids:
            triplet_pool.append((id_to_name.get(src, src), rel, id_to_name.get(tgt, tgt)))

    rng = random.Random(TRIPLET_SAMPLE_SEED)
    rng.shuffle(triplet_pool)
    triplet_samples = triplet_pool[:TRIPLET_SAMPLE_SIZE]

    return {
        "entity_types": sorted(entity_types),
        "edge_types": sorted(edge_types),
        "node_types": sorted(node_types),
        "entity_count": entity_count,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "triplet_samples": triplet_samples,
    }


async def run_approach(name: str, ontology_generation: str) -> dict:
    print(f"\n=== {name} (ONTOLOGY_GENERATION={ontology_generation}) ===")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Pruned.")

    get_cognify_config().ontology_generation = ontology_generation

    for text in JOBS:
        await cognee.add(text)
    print(f"Added {len(JOBS)} CVs.")

    TRACKER.reset()
    started = time.perf_counter()
    await cognee.cognify()
    elapsed = time.perf_counter() - started
    print(f"Cognify done in {elapsed:.1f}s.")

    stats = await collect_stats()
    print(
        f"  entities: {stats['entity_count']}, nodes: {stats['node_count']}, "
        f"edges: {stats['edge_count']}, entity types: {len(stats['entity_types'])}, "
        f"edge types: {len(stats['edge_types'])}"
    )
    print(
        f"  LLM calls: {TRACKER.calls}, prompt tokens: {TRACKER.prompt_tokens}, "
        f"completion tokens: {TRACKER.completion_tokens}, total tokens: {TRACKER.total_tokens}"
    )

    return {
        "name": name,
        "ontology_generation": ontology_generation,
        "cognify_seconds": elapsed,
        "llm_calls": TRACKER.calls,
        "prompt_tokens": TRACKER.prompt_tokens,
        "completion_tokens": TRACKER.completion_tokens,
        "total_tokens": TRACKER.total_tokens,
        **stats,
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


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_seconds(s: float) -> str:
    return f"{s:.1f}s"


def _metrics_table(results: list[dict]) -> str:
    metrics = [
        ("Cognify wall-clock time", lambda r: _fmt_seconds(r["cognify_seconds"])),
        ("LLM calls", lambda r: _fmt_int(r["llm_calls"])),
        ("Prompt tokens", lambda r: _fmt_int(r["prompt_tokens"])),
        ("Completion tokens", lambda r: _fmt_int(r["completion_tokens"])),
        ("Total tokens", lambda r: _fmt_int(r["total_tokens"])),
        ("Entities extracted", lambda r: _fmt_int(r["entity_count"])),
        ("Total nodes", lambda r: _fmt_int(r["node_count"])),
        ("Total edges", lambda r: _fmt_int(r["edge_count"])),
        ("Unique entity types", lambda r: _fmt_int(len(r["entity_types"]))),
        ("Unique edge types", lambda r: _fmt_int(len(r["edge_types"]))),
    ]
    header_cells = "<th>Metric</th>" + "".join(
        f"<th>{html.escape(r['name'])}</th>" for r in results
    )

    rows = []
    for label, accessor in metrics:
        cells = [f"<td class='label'>{html.escape(label)}</td>"]
        for r in results:
            cells.append(f"<td class='cell num'>{accessor(r)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# Compact per-batch flow diagrams (one per approach). Rendered by mermaid.js
# in the browser. Kept narrow so they fit inside each card column.
APPROACH_DIAGRAMS: dict[str, str] = {
    "Default": """flowchart TB
    A([Batch: N chunks]):::input --> E["Extract per chunk<br/>N parallel LLM calls<br/>unrestricted KnowledgeGraph"]:::par
    E --> O([KnowledgeGraph list]):::output

    classDef input fill:#dbeafe,stroke:#1e40af,color:#000
    classDef output fill:#dcfce7,stroke:#166534,color:#000
    classDef par fill:#eff6ff,stroke:#3b82f6,color:#000""",
    "AUTO_RESTRICTED": """flowchart TB
    A([Batch: N chunks]):::input
    A --> D["1 - Discover (PARALLEL)<br/>N LLM calls, one per chunk"]:::par
    D --> R["2 - Resolve UNDER LOCK<br/>2 LLM calls in parallel<br/>cluster types + relations<br/>reject narrative verbs"]:::seq
    R --> U[(self.canonical)]:::state
    U --> E["3 - Extract (PARALLEL)<br/>N LLM calls, restricted"]:::par
    E --> O([KnowledgeGraph list]):::output

    classDef input fill:#dbeafe,stroke:#1e40af,color:#000
    classDef output fill:#dcfce7,stroke:#166534,color:#000
    classDef par fill:#eff6ff,stroke:#3b82f6,color:#000
    classDef seq fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#000
    classDef state fill:#fee2e2,stroke:#991b1b,color:#000""",
    "AUTO_RESTRICTED_ITERATIVE": """flowchart TB
    A([Batch: N chunks]):::input
    A --> D["1 - Discover UNDER LOCK<br/>1 LLM call<br/>sees N chunks + prior canonical<br/>prompt: reuse existing names"]:::seq
    D --> U[(self.canonical)]:::state
    U --> E["2 - Extract (PARALLEL)<br/>N LLM calls, restricted"]:::par
    E --> O([KnowledgeGraph list]):::output

    classDef input fill:#dbeafe,stroke:#1e40af,color:#000
    classDef output fill:#dcfce7,stroke:#166534,color:#000
    classDef par fill:#eff6ff,stroke:#3b82f6,color:#000
    classDef seq fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#000
    classDef state fill:#fee2e2,stroke:#991b1b,color:#000""",
}

def _diagrams_panel(results: list[dict]) -> str:
    panels = []
    for r in results:
        diagram = APPROACH_DIAGRAMS.get(r["name"], "")
        if not diagram:
            continue
        panels.append(
            f"""
            <div class="diagram-panel">
              <h3>{html.escape(r["name"])}</h3>
              <p class="config">ONTOLOGY_GENERATION = <code>{html.escape(r["ontology_generation"])}</code></p>
              <pre class="mermaid">{diagram}</pre>
            </div>
            """
        )
    return f'<div class="diagrams">{"".join(panels)}</div>'


def _triplet_table(results: list[dict]) -> str:
    header_cells = "".join(f"<th>{html.escape(r['name'])}</th>" for r in results)
    max_rows = max((len(r["triplet_samples"]) for r in results), default=0)
    if max_rows == 0:
        return "<p class='muted-note'><em>No entity-to-entity triplets found.</em></p>"
    rows = []
    for i in range(max_rows):
        cells = []
        for r in results:
            triplets = r["triplet_samples"]
            if i < len(triplets):
                s, p, o = triplets[i]
                cells.append(
                    "<td class='triplet'>"
                    f"<span class='subj'>{html.escape(s)}</span> "
                    f"<span class='pred'>—{html.escape(p)}→</span> "
                    f"<span class='obj'>{html.escape(o)}</span>"
                    "</td>"
                )
            else:
                cells.append("<td class='triplet absent'>&mdash;</td>")
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
                <span class="pill pill-time">{_fmt_seconds(r["cognify_seconds"])}</span>
                <span class="pill pill-time">{_fmt_int(r["llm_calls"])} LLM calls</span>
                <span class="pill pill-time">{_fmt_int(r["total_tokens"])} tokens</span>
              </div>
              <div class="counts">
                <span class="pill pill-strong">{r["entity_count"]} entities</span>
                <span class="pill">{r["node_count"]} nodes</span>
                <span class="pill">{r["edge_count"]} edges</span>
              </div>
              <div class="counts">
                <span class="pill pill-alt">{len(r["node_types"])} node types</span>
                <span class="pill pill-alt">{len(r["entity_types"])} entity types</span>
                <span class="pill pill-alt">{len(r["edge_types"])} edge types</span>
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
  .pill-strong {{ background: #fef3c7; color: #92400e; }}
  .pill-alt {{ background: #f5f3ff; color: #6d28d9; }}
  .pill-time {{ background: #e0f2fe; color: #075985; }}
  pre.mermaid {{ background: transparent; margin: .35rem 0 0;
    padding: .25rem; font-size: .8rem; overflow-x: auto; }}
  pre.mermaid svg {{ max-width: 100%; height: auto; }}
  .diagrams {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
  .diagram-panel {{ border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem 1.1rem; background: #fafafa; display: flex; flex-direction: column; }}
  .diagram-panel h3 {{ margin: 0; font-size: 1.05rem; }}
  .muted-note {{ color: var(--muted); font-size: .9rem; }}
  td.triplet {{ font-size: .85rem; vertical-align: top; max-width: 320px; }}
  td.triplet .subj {{ color: #1d4ed8; font-weight: 600; }}
  td.triplet .pred {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.triplet .obj {{ color: #b91c1c; font-weight: 600; }}
  td.triplet.absent {{ color: var(--absent-fg); text-align: center; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
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
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{
    startOnLoad: true,
    securityLevel: "loose",
    flowchart: {{ htmlLabels: true, useMaxWidth: true, padding: 4 }},
    themeVariables: {{ fontSize: "11px" }}
  }});
</script>
</head>
<body>
<h1>Auto-Restricted Ontology &mdash; Approach Comparison</h1>
<p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &middot; 5 CVs ingested via simple_cognee_example</p>

<h2>Per-batch flow &mdash; how each approach works</h2>
{_diagrams_panel(results)}

<h2>Graph metrics &mdash; at a glance</h2>
{_metrics_table(results)}

<h2>Per-approach results</h2>
<div class="cards">{"".join(cards)}</div>

<h2>Sample triplets (random {TRIPLET_SAMPLE_SIZE} entity&ndash;entity edges)</h2>
{_triplet_table(results)}

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
