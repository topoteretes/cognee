"""Ingest a Mixpanel project's analytics definitions into cognee memory, incrementally.

Demonstrates the Mixpanel connector: each annotation, cohort definition, saved funnel
report and Lexicon event schema becomes one text document, so cognify can answer
questions like "what does the team track as a signup?" or "which cohorts exist for
power users". Re-running the script lists every definition again but re-ingests only
what changed, and definitions deleted in Mixpanel are forgotten automatically.

Definitions are knowledge; raw events are noise, so this example syncs definitions
only. Pass include=("events",) and events_from="YYYY-MM-DD" to also ingest daily
per-event counts over a date window (see the README).

Setup:
  - Create a service account in Mixpanel under Organization Settings > Service
    Accounts with access to your project, and copy its username and secret.
  - Set:
      MIXPANEL_SERVICE_ACCOUNT_USERNAME=<service account username>
      MIXPANEL_SERVICE_ACCOUNT_SECRET=<service account secret>
  - If the service account sees more than one project, also set MIXPANEL_PROJECT_ID.

Also set the usual cognee LLM_API_KEY (see cognee's .env.template) — this example
calls cognee.recall(), which needs an LLM for the final completion.

Install:
    pip install "cognee-connector-mixpanel[cognee]"

Run it:

    python connectors/mixpanel/examples/mixpanel.py
"""

import asyncio
import os

import cognee

from cognee_connector_mixpanel import mixpanel_source

DATASET_NAME = "mixpanel_demo"
RESOURCE_NAME = "mixpanel_product"

# "merge" is required: it makes re-runs incremental and propagates deleted
# definitions through orphan cleanup.
REMEMBER_KWARGS = {
    "dataset_name": DATASET_NAME,
    "primary_key": "id",
    "write_disposition": "merge",
}


def build_source():
    return mixpanel_source(
        resource_name=RESOURCE_NAME,
        service_account_username=os.environ["MIXPANEL_SERVICE_ACCOUNT_USERNAME"],
        service_account_secret=os.environ["MIXPANEL_SERVICE_ACCOUNT_SECRET"],
        project_id=os.environ.get("MIXPANEL_PROJECT_ID") or None,
        region=os.environ.get("MIXPANEL_REGION", "us"),
    )


async def main():
    print("=== Initial sync ===")
    source = build_source()
    await cognee.remember(source, **REMEMBER_KWARGS)
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(
        "Which Mixpanel cohorts exist and what do they track?",
        datasets=[DATASET_NAME],
    )
    print("Recall:", answer[0].text if answer else "nothing found")

    print("\n=== Incremental re-sync (only new, changed or deleted definitions) ===")
    source = build_source()
    await cognee.remember(source, **REMEMBER_KWARGS)
    print("Sync stats:", source.cognee_sync_stats)


if __name__ == "__main__":
    asyncio.run(main())
