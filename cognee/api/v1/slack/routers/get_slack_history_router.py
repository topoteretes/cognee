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
        - **team_id** (str): Slack workspace (team) id to import from; starting a second
          import for the same workspace and dataset while one is running fails with 400.

        ## Request Parameters
        - **channel_ids** (List[str]): Slack channel IDs (C… or G…) to import; channel names
          and DMs are rejected and duplicates are dropped. Channel imports also require days
          or oldest, and at least one channel or thread must be selected. At most 50 items.
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        - **days** (Optional[int]): Length of the look-back window ending now; combining it
          with oldest or latest is rejected. A thread-only selection that gives no bound at
          all falls back to 7 days. Must be between 1 and 3650.
        - **latest** (Optional[datetime]): End of the date window; must carry a timezone,
          must follow oldest, cannot be combined with days, and cannot be later than now.
          Defaults to the current time when omitted.
        - **max_messages** (int): Cap on the messages fetched for this import; reaching the
          cap fails the import instead of returning a truncated snapshot. Must be between 1
          and 500000. Defaults to 50000.
        - **max_requests** (int): Cap on the Slack API requests this import may issue;
          reaching the cap fails the import instead of returning a truncated snapshot. Must
          be between 1 and 10000. Defaults to 1000.
        - **oldest** (Optional[datetime]): Start of the date window; must carry a timezone
          and precede latest, and cannot be combined with days. Channel selections need
          either this or days.
        - **thread_links** (List[str]): Slack message permalinks
          (https://<workspace>.slack.com/archives/…) selecting whole conversations
          regardless of the date window; duplicates are dropped and links of any other shape
          are rejected. At most 50 items.
        - **thread_mode** (Literal['started', 'active']): 'started' selects threads whose
          root falls inside the window; 'active' also scans older roots and selects threads
          with any message in the window. Neither acts as a retention policy. One of:
          'started', 'active'. Defaults to 'started'.
        - **threads** (List[SlackThread]): Channel id and thread timestamp pairs naming
          conversations to import; like thread links, a selected thread brings in all of its
          available replies. At most 50 items.
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
        """Configure or disable the recurring Slack history sync for a dataset.

        Returns 400 when the settings or selection are invalid, when no selection is
        supplied, or when a caller who does not own the Slack connection tries to disable
        its sync, and 403 when the caller lacks read, write and delete access to the target
        dataset. The stored settings are returned with days resolved once into a fixed
        oldest date, so the window boundary does not move between runs.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id whose per-dataset sync settings are
          written; disabling a sync is allowed only for the user who owns the Slack
          connection.

        ## Request Parameters
        - **enabled** (bool): When true, the selection is refreshed on the given interval;
          enabling without a selection is rejected. Disabling still needs selection to name
          the dataset and is allowed only to the connection owner. Defaults to False.
        - **interval_seconds** (int): Time between refresh runs of the stored selection.
          Must be between 600 and 604800. Defaults to 21600.
        - **selection** (Optional[SlackHistoryRequest]): Import selection to refresh,
          including the dataset it targets; required even when disabling, since it names the
          dataset. A fixed latest date is rejected and days is resolved once into a stored
          oldest date.
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
        """Report stored sync settings and import reports for a Slack workspace.

        Returns 404 when the workspace has no active Slack connection owned by the caller.
        The body has sync (stored sync settings keyed by dataset id), reports (per-dataset
        import reports with status, started_at, finished_at and either result or an error
        message) and worker_enabled; sync and reports are empty objects when nothing has
        been configured or run.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id to report on; a workspace that is not
          connected, no longer active, or connected by another user returns 404.
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
