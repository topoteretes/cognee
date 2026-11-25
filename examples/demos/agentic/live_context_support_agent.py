"""Answer a support chat with live behavior from Snowplow Signals and long-term memory from cognee.

Maya bought Talusworks Ridgeline trail shoes in the Autumn Sale, in her usual EU 39, and they are
too small. Snowplow Signals supplies what she is doing right now; cognee remembers what was said and
written before: her orders, the returns policy, and the store's fit notes, which never mention her.

Scene 1: a Signals rule fires while she keeps opening the size guide and the returns page, and the
agent opens the chat before she types. She asks to return the shoes. Sale items are final sale, so
the agent offers a free exchange instead, one size up, because the fit notes say Ridgeline runs a
size small. The chat outcome and a summary of her on-site behavior are then remembered.

Scene 2, two weeks later: she is looking at Northpeak Summit Hiker shoes and asks which size to
take. Signals only knows what she is looking at; cognee remembers that she wears 39 and needed a 40
only because the Ridgeline runs small, and the fit notes say the Summit Hiker runs true to size. So:
take the 39.

Each scene answers the chat twice with the same recall: once over the store's node set only
("Signals only": live behavior plus the store's documents, no memory of Maya), then with Maya's
node set added ("Signals + cognee"). The live behavior goes into the query text.

Signals data comes from the samples in live_context_support_agent_data/ unless
SNOWPLOW_SIGNALS_API_URL is set; then the scenes read the visitor's real attributes (and, with
SNOWPLOW_SIGNALS_EVENT_LOG, the event log narrative remembered after scene 1) with the
snowplow-signals SDK (`uv run --with snowplow-signals python ...`), which also needs
SNOWPLOW_SIGNALS_API_KEY, SNOWPLOW_SIGNALS_API_KEY_ID, SNOWPLOW_SIGNALS_ORG_ID and
SNOWPLOW_SIGNALS_SERVICE (optional: SNOWPLOW_SIGNALS_ATTRIBUTE_KEY, SNOWPLOW_SIGNALS_IDENTIFIER).

Requires: LLM_API_KEY. Customers are separated with node sets, which need Ladybug (the default
graph store) or Neo4j.
Run: uv run python examples/demos/agentic/live_context_support_agent.py
"""

# ruff: noqa: E402
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# Set os.environ before importing Cognee: Cognee reads env-backed settings at import time.
os.environ.setdefault("LOG_LEVEL", "ERROR")

import cognee

DATASET = "live_context_support_agent_demo"
CUSTOMER_ID = "maya-7f3a"
STORE_NODE_SET = "store"

# Sample inputs live in the sibling data folder:
# - customer orders, remembered in each customer's node set, and the returns policy and fit notes,
#   remembered together as one document in the shared store node set, before scene 1
# - the scene 1 chat outcome and a behavior narrative shaped like Signals' get_agentic_context(...,
#   format="narrative"), remembered after it
# - Signals attribute snapshots shaped like get_service_attributes: four for scene 1 as Maya browses,
#   one for scene 2
DATA_DIR = Path(__file__).parent / "live_context_support_agent_data"
MAYA_ORDERS = (DATA_DIR / "maya_orders.txt").read_text()
LEO_ORDERS = (DATA_DIR / "leo_orders.txt").read_text()
RETURNS_POLICY = (DATA_DIR / "returns_policy.txt").read_text()
FIT_NOTES = (DATA_DIR / "fit_notes.txt").read_text()
CHAT_OUTCOME = (DATA_DIR / "chat_outcome.txt").read_text()
SAMPLE_BEHAVIOR_NARRATIVE = (DATA_DIR / "behavior_narrative.txt").read_text()
SNAPSHOTS = json.loads((DATA_DIR / "signals_snapshots.json").read_text())


def signals_client():
    from snowplow_signals import Signals

    return Signals(
        api_url=os.environ["SNOWPLOW_SIGNALS_API_URL"],
        api_key=os.environ["SNOWPLOW_SIGNALS_API_KEY"],
        api_key_id=os.environ["SNOWPLOW_SIGNALS_API_KEY_ID"],
        org_id=os.environ["SNOWPLOW_SIGNALS_ORG_ID"],
    )


def get_live_context(customer_id: str, sample: dict) -> dict:
    """Return the shopper's live attributes from Snowplow Signals, or the sample snapshot."""
    if not os.getenv("SNOWPLOW_SIGNALS_API_URL"):
        return sample
    return signals_client().get_service_attributes(
        name=os.environ["SNOWPLOW_SIGNALS_SERVICE"],
        attribute_key=os.getenv("SNOWPLOW_SIGNALS_ATTRIBUTE_KEY", "domain_userid"),
        identifier=os.getenv("SNOWPLOW_SIGNALS_IDENTIFIER", customer_id),
    )


def get_behavior_narrative(customer_id: str) -> str:
    """Return the session's behavior from the Signals event log as text, or the sample narrative."""
    if not os.getenv("SNOWPLOW_SIGNALS_API_URL") or not os.getenv("SNOWPLOW_SIGNALS_EVENT_LOG"):
        return SAMPLE_BEHAVIOR_NARRATIVE
    return signals_client().get_agentic_context(
        name=os.environ["SNOWPLOW_SIGNALS_EVENT_LOG"],
        identifier=os.getenv("SNOWPLOW_SIGNALS_IDENTIFIER", customer_id),
        format="narrative",
    )


def wants_help_with_return(live: dict) -> bool:
    """The trigger rule. In production this is a Signals intervention that fires on the site."""
    return (live.get("returns_page_views_10m") or 0) >= 3


# --- The support agent --------------------------------------------------------------------------

AGENT_PROMPT = (
    "You are the support chat agent for Northpeak Outfitters, an online outdoor gear store. "
    "Answer the customer in two to four friendly sentences. Use only the context you are given; "
    "if you lack the details to answer, ask for them. Check every option you offer against the "
    "policy and the customer's purchase details."
)


def describe(live: dict) -> str:
    return "Live behavior from Snowplow Signals:\n" + json.dumps(live, indent=2)


async def compare_answers(message: str, live: dict, today: str):
    """Answer the chat message with Signals only, then with Signals + cognee."""
    print(f'\nMaya: "{message}"')
    query = (
        f"Today is {today}. Customer {CUSTOMER_ID} is on our site. {describe(live)}\n"
        f'They wrote: "{message}"'
    )
    # The same recall twice: without memory of the customer (only the store's documents), then
    # with it. A fresh session_id keeps answers cached in the default session out of the context.
    for label, node_sets in [
        ("Signals only (live behavior + the store's documents)", [STORE_NODE_SET]),
        (
            "Signals + cognee (live behavior + the store's documents + Maya's memory)",
            [STORE_NODE_SET, CUSTOMER_ID],
        ),
    ]:
        result = await cognee.recall(
            query_text=query,
            datasets=[DATASET],
            node_name=node_sets,
            system_prompt=AGENT_PROMPT,
            session_id=f"support-{uuid4().hex[:8]}",
        )
        print(f"\n--- {label} ---\n{result[0].text}")


async def run():
    print("Resetting the demo dataset and building memory...")
    if any(dataset.name == DATASET for dataset in await cognee.datasets.list_datasets()):
        await cognee.forget(dataset=DATASET)

    # Each customer's documents go in their own node set; the store's documents in a shared one.
    await cognee.remember(MAYA_ORDERS, dataset_name=DATASET, node_set=[CUSTOMER_ID])
    await cognee.remember(LEO_ORDERS, dataset_name=DATASET, node_set=["leo-21bd"])
    await cognee.remember(
        f"{RETURNS_POLICY}\n\n{FIT_NOTES}", dataset_name=DATASET, node_set=[STORE_NODE_SET]
    )

    print("\n=== Scene 1, Oct 26: Maya's sale trail shoes are too small ===")
    for step, sample in enumerate(SNAPSHOTS["scene_1"], start=1):
        live = get_live_context(CUSTOMER_ID, sample)
        print(f"[signals] snapshot {step}: {json.dumps(live)}")
        if wants_help_with_return(live):
            print("[signals] trigger fired: returns_page_views_10m >= 3, the agent opens the chat")
            opener = await cognee.recall(
                query_text=(
                    f"Today is Oct 26. Customer {CUSTOMER_ID} is on our site. {describe(live)}\n"
                    "They have not written anything yet. Open the chat proactively about what "
                    "they seem to need, in two sentences."
                ),
                datasets=[DATASET],
                node_name=[STORE_NODE_SET, CUSTOMER_ID],
                system_prompt=AGENT_PROMPT,
                session_id=f"support-{uuid4().hex[:8]}",
            )
            print(f"\nAgent (unprompted): {opener[0].text}")
            break
    await compare_answers(
        "These are too small, I'd like to return them. I normally wear a 39.", live, "Oct 26"
    )

    print("\nRemembering the chat outcome and Maya's on-site behavior...")
    await cognee.remember(
        [CHAT_OUTCOME, get_behavior_narrative(CUSTOMER_ID)],
        dataset_name=DATASET,
        node_set=[CUSTOMER_ID],
    )

    print("\n=== Scene 2, Nov 9: Maya is back, looking at a different shoe ===")
    live = get_live_context(CUSTOMER_ID, SNAPSHOTS["scene_2"])
    print(f"[signals] snapshot: {json.dumps(live)}")
    await compare_answers("Which size should I get in these?", live, "Nov 9")


if __name__ == "__main__":
    asyncio.run(run())
