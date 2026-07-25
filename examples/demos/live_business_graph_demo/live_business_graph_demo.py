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
# The relational engine won't create missing parent directories, and the
# demo's first DB touch (agent registration) comes before any ingestion
# path that would create them.
os.makedirs(os.path.join(DEMO_ROOT, "data"), exist_ok=True)
os.makedirs(os.path.join(DEMO_ROOT, "system", "databases"), exist_ok=True)

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

SLACK_MESSAGES = """Slack digest — #go-to-market channel.
Priya Nair: the "Retail Reimagined" campaign is live — targeting retail
companies through LinkedIn and the email newsletter. It already generated
the Acme Retail opportunity and 40 other leads.
Priya Nair: the "Ops Excellence" webinar series ran on the events channel
and brought in Nordwind Logistics.
Both campaigns promote the Insights Dashboard product."""

DRIVE_DOCS = """Google Drive — support and product usage review doc.
Tom Becker from Acme Retail filed ticket #4821 about slow report exports
in the Insights Dashboard; it was resolved by engineer Milan Kovac.
Usage analytics show Acme Retail runs 300 dashboard queries daily, while
Nordwind Logistics adoption dropped 20% after the ticket #5177 escalation.
Ines Vogel gave feedback that the alerting feature drives their renewal
decision. Renewal revenue at risk: 80k EUR."""

SOURCES = [
    ("CRM (accounts & contacts)", CRM_NOTES, "crm"),
    ("Slack (#go-to-market)", SLACK_MESSAGES, "slack"),
    ("Google Drive (usage reviews)", DRIVE_DOCS, "google_drive"),
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


AGENT_NAME = "business-copilot"


async def connect_agent(with_dataset: bool) -> None:
    """Register the demo agent connection — a REAL registry entry, so the
    Agent node on the page comes from the live agents subsystem, not a prop."""
    await cognee.agents.register(
        AGENT_NAME,
        type="mcp",
        source="mcp",
        memory_mode="cognee",
        # Its OWN session: phase-4 questions run under it, so the agent card
        # shows the agent's conversation history, not the operator's.
        session_id="copilot-session",
        dataset_names=[DATASET] if with_dataset else None,
        origin_function="live_business_graph_demo",
    )


async def main() -> None:
    # Fresh store: create the relational schema before anything touches it
    # (agent registration reads the users table before any ingestion runs).
    from cognee.modules.engine.operations.setup import setup

    await setup()

    banner("PHASE 0 · Agent connected")
    await connect_agent(with_dataset=False)
    print(f"   Agent connection '{AGENT_NAME}' registered (type=mcp). No data yet.")

    for index, (label, text, node_set) in enumerate(SOURCES, start=1):
        banner(f"PHASE {index} · Plugging in data source: {label}")
        await cognee.add(text, dataset_name=DATASET, node_set=[node_set])
        print("   ingesting…")
        await cognee.cognify([DATASET])
        print("   entities and relationships extracted.")
        if index == 1:
            # The dataset now exists — scope the agent connection to it so the
            # agent —reads/writes→ dataset edges appear on the page.
            await connect_agent(with_dataset=True)
            print(f"   agent '{AGENT_NAME}' scoped to dataset '{DATASET}'.")
        await render_page()
        if index == 1:
            print(f"\n   >>> OPEN {PAGE_PATH} IN A BROWSER NOW <<<")
            print("   (it will keep refreshing itself as sources are added)")
        print(f"   watch the page refresh — pausing {PAUSE_SECONDS:.0f}s…")
        await asyncio.sleep(PAUSE_SECONDS)

    banner("PHASE 3.5 · The organization around the data")
    # A colleague with her own agent, a read grant on the business dataset,
    # and a separate body of organizational knowledge — so the page shows
    # who owns what and who can access what, not just a single user.
    from cognee.modules.users.methods import create_user, get_default_user
    from cognee.modules.users.permissions.methods import give_permission_on_dataset
    from cognee.modules.data.methods import get_datasets_by_name

    owner = await get_default_user()
    try:
        maya = await create_user(email="maya@novagraph.dev", password="demo-Maya-1")
        print("   colleague maya@novagraph.dev created.")
    except Exception:
        from cognee.modules.users.methods import get_user_by_email

        maya = await get_user_by_email("maya@novagraph.dev")
        print("   colleague maya@novagraph.dev already exists.")
    business = (await get_datasets_by_name(DATASET, owner.id))[0]
    for permission in ("read", "share"):
        try:
            await give_permission_on_dataset(maya, business.id, permission)
        except Exception:
            pass  # re-runs: grant already exists
    print(f"   maya granted read+share on '{DATASET}'.")
    await cognee.agents.register(
        "support-analyst",
        user=maya,
        type="sdk",
        source="api",
        memory_mode="session",
        # By id, not name: name lookup only finds datasets the user OWNS,
        # and maya has a grant on this one, not ownership.
        dataset_ids=[str(business.id)],
        session_id="analyst-session",
        origin_function="live_business_graph_demo",
    )
    print("   maya's agent 'support-analyst' registered and scoped to the dataset.")
    await cognee.search(
        query_text="Which support tickets affected customer adoption?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[business.id],
        top_k=8,
        session_id="analyst-session",
        user=maya,
    )
    print("   support-analyst asked its first question in its own session.")

    await cognee.add(
        "Personal notes: my working playbook. Mission focus is connected customer knowledge. "
        "Escalation policy: enterprise tickets route to the on-call engineer within "
        "one hour. Renewal playbook: accounts with falling adoption get an executive "
        "sponsor call.",
        dataset_name="personal_notes",
        node_set=["org-knowledge"],
    )
    await cognee.cognify(["personal_notes"])
    print("   personal notes added as their own dataset (personal_notes).")

    # The TEAM BRAIN: shared working knowledge both people read and write —
    # decisions, conventions, learnings that belong to the team, not a person.
    await cognee.add(
        "Team decisions log. Decision: Acme Retail renewal is our top Q3 priority, "
        "owner Sara Lopez. Convention: every escalated ticket gets a post-mortem "
        "note. Learning: campaigns convert best when paired with a webinar. "
        "Decision: Nordwind Logistics gets an executive sponsor call in July.",
        dataset_name="team_brain",
        node_set=["team-decisions"],
    )
    await cognee.cognify(["team_brain"])
    team_ds = (await get_datasets_by_name("team_brain", owner.id))[0]
    for permission in ("read", "write"):
        try:
            await give_permission_on_dataset(maya, team_ds.id, permission)
        except Exception:
            pass
    print("   team brain created and shared with the whole team (read+write).")
    await render_page()
    print(
        f"   watch the page — operators and knowledge panels now show the org. Pausing {PAUSE_SECONDS:.0f}s…"
    )
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
            session_id="copilot-session",
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
