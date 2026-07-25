"""Live business-graph demo — fragmented sources become one connected model.

The story this tells (keep the visualization page open and watch):

  1. An "agent" is connected to cognee.
  2. Business data sources are plugged in one at a time — CRM, marketing
     campaigns, support + product usage.
  3. After each source, cognee identifies people, customers, campaigns,
     channels and products, and the open page REFRESHES ITSELF to show the
     new entities and the relationships forming across sources.
  4. The agent then asks business questions — each answer's supporting
     subgraph SPOTLIGHTS LIVE on the page (Memory tab).

The point for the audience: not tables and schemas — entities of their
business, connecting up as data arrives, then being *used*.

Run it:

    # terminal 1 — the cognee API server (serves live events to the page)
    ENABLE_BACKEND_ACCESS_CONTROL=false CORS_ALLOWED_ORIGINS="null" \
    DATA_ROOT_DIRECTORY=/tmp/cognee_biz_demo/data \
    SYSTEM_ROOT_DIRECTORY=/tmp/cognee_biz_demo/system \
        uvicorn cognee.api.client:app --port 8000

    # terminal 2 — this driver
    python examples/demos/live_business_graph_demo/live_business_graph_demo.py

    # browser — open /tmp/business_graph.html after phase 1 renders it
    # (the page then keeps itself up to date)

Requires a working LLM_API_KEY. All state lives under /tmp/cognee_biz_demo.
"""

import asyncio
import os

DEMO_ROOT = os.environ.get("DEMO_ROOT", "/tmp/cognee_biz_demo")
# Force-set (not setdefault): shell profiles commonly export these pointing at
# another cognee checkout, and the demo must run against its own isolated
# store — sharing a relational DB across checkouts breaks alembic migrations.
os.environ["DATA_ROOT_DIRECTORY"] = os.path.join(DEMO_ROOT, "data")
os.environ["SYSTEM_ROOT_DIRECTORY"] = os.path.join(DEMO_ROOT, "system")
os.environ.setdefault("TELEMETRY_DISABLED", "1")

import cognee  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402

DATASET = "business"
PAGE_PATH = os.environ.get("DEMO_PAGE", "/tmp/business_graph.html")
PAUSE_SECONDS = float(os.environ.get("DEMO_PAUSE", "12"))

# ── The fragmented sources ──────────────────────────────────────────────
# Deliberately overlapping: the same customers, products and people appear
# in different systems, so cross-source relationships form on the graph.

CRM_NOTES = """CRM export — accounts and contacts.
Acme Retail is an enterprise customer, account owner Sara Lopez.
Their champion is Tom Becker, Head of E-commerce at Acme Retail.
Nordwind Logistics is a mid-market customer, account owner Sara Lopez.
Nordwind's main contact is Ines Vogel, VP Operations.
Both accounts use the Insights Dashboard product."""

MARKETING_CAMPAIGNS = """Marketing report — Q2 campaigns.
The "Retail Reimagined" campaign targeted retail companies through the
LinkedIn channel and the email newsletter. It generated the Acme Retail
opportunity and 40 other leads.
The "Ops Excellence" webinar series ran on the events channel and brought
in Nordwind Logistics. Campaign manager for both is Priya Nair.
Both campaigns promote the Insights Dashboard product."""

SUPPORT_AND_USAGE = """Support and product usage digest.
Tom Becker from Acme Retail filed ticket #4821 about slow report exports
in the Insights Dashboard; it was resolved by engineer Milan Kovac.
Usage analytics show Acme Retail runs 300 dashboard queries daily, while
Nordwind Logistics adoption dropped 20% after the ticket #5177 escalation.
Ines Vogel gave feedback that the alerting feature drives their renewal
decision. Renewal revenue at risk: 80k EUR."""

SOURCES = [
    ("CRM (accounts & contacts)", CRM_NOTES, "crm"),
    ("Marketing (campaigns & channels)", MARKETING_CAMPAIGNS, "marketing"),
    ("Support + product usage", SUPPORT_AND_USAGE, "support_usage"),
]

# ── The agent's questions ───────────────────────────────────────────────

AGENT_QUESTIONS = [
    "How are our marketing campaigns connected to customers and products?",
    "Which customer relationships are at risk and why?",
    "Who should talk to Tom Becker about his experience, and what happened?",
]


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


async def render_page() -> None:
    await cognee.visualize_graph(
        destination_file_path=PAGE_PATH,
        dataset=DATASET,
        full=True,
        live=True,
    )
    print(f"   page rendered → {PAGE_PATH} (open it now if you haven't)")


async def main() -> None:
    banner("PHASE 0 · Agent connected")
    print("   The agent is wired to cognee (search API / MCP). No data yet.")

    for index, (label, text, node_set) in enumerate(SOURCES, start=1):
        banner(f"PHASE {index} · Plugging in data source: {label}")
        await cognee.add(text, dataset_name=DATASET, node_set=[node_set])
        print("   ingesting…")
        await cognee.cognify([DATASET])
        print("   entities and relationships extracted.")
        await render_page()
        if index == 1:
            print(f"\n   >>> OPEN {PAGE_PATH} IN A BROWSER NOW <<<")
            print("   (it will keep refreshing itself as sources are added)")
        print(f"   watch the page refresh — pausing {PAUSE_SECONDS:.0f}s…")
        await asyncio.sleep(PAUSE_SECONDS)

    banner("PHASE 4 · The agent uses the connected business context")
    print("   Switch the page to the Memory tab — each answer's subgraph")
    print("   spotlights as the agent asks:")
    for question in AGENT_QUESTIONS:
        print(f"\n   ⌕ {question}")
        results = await cognee.search(
            query_text=question,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[DATASET],
            top_k=10,
        )
        answer = ""
        if results and isinstance(results[0], dict):
            payload = results[0].get("search_result") or [""]
            answer = payload[0] if payload else ""
        elif results:
            answer = str(results[0])
        print(f"     → {answer[:300]}")
        await asyncio.sleep(PAUSE_SECONDS / 2)

    banner("Done — fragmented sources are now one connected business model.")


if __name__ == "__main__":
    asyncio.run(main())
