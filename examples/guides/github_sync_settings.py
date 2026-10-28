"""Configure a connected GitHub App for weekday sync and selected code paths.

Run with --account-id INSTALLATION_ID. By default this only previews the schedule;
--apply saves it, and --acknowledge-removal explicitly authorizes replacing the
previous repository/file selection. No scheduler service is started by this script.
"""

import argparse
import asyncio
from datetime import datetime, timezone

from cognee.modules.integrations.credentials import get_active_credential_for_user
from cognee.modules.integrations.github import (
    GithubRepositorySettings,
    GithubSyncSettings,
    SyncSchedule,
    next_sync_at,
    save_sync_settings,
)
from cognee.modules.users.methods import get_default_user


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--repository-id", required=True, help="Numeric GitHub repository id")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--acknowledge-removal", action="store_true")
    args = parser.parse_args()
    settings = GithubSyncSettings(
        mode="scheduled",
        first_sync="scheduled",
        schedule=SyncSchedule(
            interval_minutes=15,
            weekdays=[0, 1, 2, 3, 4],
            timezone="Europe/Belgrade",
            start_time="09:00",
            end_time="17:00",
        ),
        repositories={
            args.repository_id: GithubRepositorySettings(
                branch="main",
                include_paths=["src/**"],
                exclude_paths=["**/tests/**", "**/generated/**"],
            )
        },
    )
    print(settings.model_dump_json(indent=2))
    print("Next eligible run:", next_sync_at(settings, datetime.now(timezone.utc)))
    if args.apply:
        user = await get_default_user()
        credential = await get_active_credential_for_user(user.id, "github", args.account_id)
        if credential is None:
            raise ValueError("Connect this GitHub installation for the current user first")
        await save_sync_settings(credential, settings, acknowledge_removal=args.acknowledge_removal)
        print("Saved. Invoke run_due_github_syncs() from your scheduler to execute due runs.")


if __name__ == "__main__":
    asyncio.run(main())
