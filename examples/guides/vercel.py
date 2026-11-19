"""Vercel connector demo: ask memory which deploy failed and why.

``vercel_source`` returns a ``dlt`` resource that you hand straight to ``cognee.remember``. One
source is one Vercel project: the project and each of its deployments become one document, with
the environment, state, branch, commit sha, commit message, author login and error code in the
first lines. A re-sync extracts only the deployments whose text changed, and a deployment
Vercel deletes leaves memory on the next sync (it is never forgotten just for being old).

This script syncs one project, asks a question, syncs again (nothing is extracted twice) and
asks again. Run it after a failed deploy to see the error in the answer.

Privacy / opt-in
----------------
Project settings are read by name only: environment variables, deploy hooks, protection bypass
keys, trusted IPs, build commands and every email address are never copied. Build output is off
by default (``include_build_logs=True`` turns it on): a build prints whatever it prints, and
Vercel redacts only secret values of 32 characters or more. What is read goes to your configured
LLM and embedding providers. Keep it in its own dataset so it is easy to inspect and forget
(``cognee.forget`` does not remove the staged copy, so the next sync builds the documents again).

Setup
-----
1. Create a token at https://vercel.com/account/tokens, scoped to the team, with an expiry.
2. Find the project id (``prj_...``) under the project's Settings, General.
3. Set your LLM key (``LLM_API_KEY``) in ``.env`` like any other cognee example, then:

       export VERCEL_TOKEN="..."
       export VERCEL_PROJECT_ID="prj_..."
       export VERCEL_TEAM_ID="team_..."      # or VERCEL_ACCOUNT_ID=<your user id> on Hobby
       uv run python examples/guides/vercel.py
"""

import asyncio
import os

import cognee
from cognee.tasks.ingestion.connectors import vercel_source

# One dataset per Vercel team (or personal account), so it is easy to inspect and forget.
DATASET_NAME = "vercel_demo"

QUESTION = "Which production deployment failed most recently, with what error, and which commit?"


async def sync_and_ask() -> None:
    project_id = os.environ["VERCEL_PROJECT_ID"]
    source = vercel_source(
        # One name per user and project: it names the staging table and the sync state.
        resource_name=f"vercel_{project_id.lower()}",
        project_id=project_id,
        team_id=os.environ.get("VERCEL_TEAM_ID"),
        account_id=os.environ.get("VERCEL_ACCOUNT_ID"),
        access_token=os.environ["VERCEL_TOKEN"],
    )
    # Pass the same arguments on every later sync.
    await cognee.remember(
        source,
        dataset_name=DATASET_NAME,
        primary_key="id",
        # REQUIRED: the add pipeline defaults to "replace", which would wipe the synced
        # project on the next sync.
        write_disposition="merge",
        # Read the whole staging table back, so a deletion is seen.
        max_rows_per_table=0,
        self_improvement=False,
    )
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(
        query_text=QUESTION,
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print(f"Q: {QUESTION}\nA: {answer}\n")


async def main() -> None:
    await sync_and_ask()
    # Nothing changed at Vercel: every row is skipped and nothing is extracted again.
    await sync_and_ask()


if __name__ == "__main__":
    asyncio.run(main())
