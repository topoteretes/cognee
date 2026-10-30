"""Authenticated Slack history selection and sync controls."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from cognee.modules.integrations.slack.history_models import (
    SlackHistoryError,
    SlackHistoryRequest,
    SlackHistoryResult,
    SlackSyncSettings,
)
from cognee.modules.integrations.slack.history_sync import (
    configure_slack_sync,
    run_history_import,
    slack_history_lifespan,
)
from cognee.modules.integrations.slack.persistence import get_by_team, is_active
from cognee.modules.integrations.slack.slack_settings import slack_settings
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


def get_slack_history_router():
    router = APIRouter(lifespan=slack_history_lifespan)

    @router.post("/history/{team_id}/import", response_model=SlackHistoryResult)
    async def import_history(
        team_id: str,
        selection: SlackHistoryRequest,
        user: Annotated[User, Depends(get_authenticated_user)],
    ):
        """Fetch and index the selection; returns only after native indexing succeeds.

        This is a long-running request. For interactive use, the Slack import
        dialog acknowledges immediately and reports its background result.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id of the connection to import from.

        ## Request Parameters
        - **channel_ids** (List[str]): Slack channel IDs (C... or G...) to import, not
          channel names or DMs; duplicates are dropped. Channel imports also require days or
          an oldest date. At most 50 items.
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        - **days** (Optional[int]): Length of the import window ending now, in days. Cannot
          be combined with oldest or latest; channel imports need either this or an oldest
          date. Must be between 1 and 3650.
        - **latest** (Optional[datetime]): End of the import window, timezone-aware and
          later than oldest. Cannot be combined with days; defaults to the time of the
          request.
        - **max_messages** (int): Cap on messages fetched for this import. Reaching the cap
          fails the import rather than returning a truncated snapshot. Must be between 1 and
          500000. Defaults to 50000.
        - **max_requests** (int): Cap on Slack API requests made for this import. Reaching
          the cap fails the import rather than returning a truncated snapshot. Must be
          between 1 and 10000. Defaults to 1000.
        - **oldest** (Optional[datetime]): Start of the import window, timezone-aware and
          earlier than latest. Cannot be combined with days.
        - **thread_links** (List[str]): Slack message or thread permalinks
          (https://<workspace>.slack.com/archives/...) selecting whole conversations
          regardless of the date window; duplicates are dropped. At most 50 items.
        - **thread_mode** (Literal['started', 'active']): How the date window selects
          threads: 'started' takes thread roots inside the window, 'active' also scans older
          roots and takes threads with a message in the window. One of: 'started', 'active'.
          Defaults to 'started'.
        - **threads** (List[SlackThread]): Threads named directly by channel id and root
          timestamp, selected with all available replies. At most 50 items.
        """
        try:
            return await run_history_import(team_id, selection, user=user)
        except SlackHistoryError as error:
            raise HTTPException(400, str(error)) from error
        except PermissionDeniedError as error:
            raise HTTPException(
                403, "Read, write and delete access to the target dataset is required."
            ) from error

    @router.put("/history/{team_id}/sync", response_model=SlackSyncSettings)
    async def set_sync(
        team_id: str,
        settings: SlackSyncSettings,
        user: Annotated[User, Depends(get_authenticated_user)],
    ):
        """Configure the recurring Slack history sync for a dataset.

        Returns the stored settings with the selection's date window resolved to a fixed
        oldest date. Returns 400 for an invalid or missing selection, and 403 when enabling
        without read, write and delete access to the target dataset.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id whose sync configuration is being
          set.

        ## Request Parameters
        - **enabled** (bool): Turns the recurring sync on for the selection's dataset.
          Enabling requires a selection and read, write and delete access to its dataset;
          disabling is permitted for the owner of the Slack connection. Defaults to False.
        - **interval_seconds** (int): Seconds to wait after a completed sync of this
          dataset before the next one. Must be between 600 and 604800. Defaults to 21600.
        - **selection** (Optional[SlackHistoryRequest]): Import selection identifying the
          dataset to configure and what to sync; required, and 400 is returned without it.
          Its latest date must be unset, and a relative days window is resolved once into a
          fixed oldest date when stored.
        """
        try:
            return await configure_slack_sync(team_id, settings, user=user)
        except (SlackHistoryError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        except PermissionDeniedError as error:
            raise HTTPException(
                403, "Read, write and delete access to the target dataset is required."
            ) from error

    @router.get("/history/{team_id}")
    async def history_status(team_id: str, user: Annotated[User, Depends(get_authenticated_user)]):
        """Read sync settings and import reports for a Slack workspace.

        Returns the stored sync configuration, per-dataset import reports, and whether the
        history sync worker is enabled. Returns 404 when the caller has no active connection
        for the workspace.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id whose sync settings and import
          reports are returned.
        """
        credential = await get_by_team(team_id)
        if not is_active(credential) or credential.user_id != user.id:
            raise HTTPException(404, "Slack connection not found.")
        metadata = credential.provider_metadata or {}
        return {
            "sync": metadata.get("history_sync") or {},
            "reports": metadata.get("history_reports") or {},
            "worker_enabled": bool(
                slack_settings.history_sync_enabled and slack_settings.client_id
            ),
        }

    return router
