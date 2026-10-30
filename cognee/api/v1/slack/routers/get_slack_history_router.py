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
        - **team_id** (str): Slack workspace whose connection supplies the messages. Only
          one import per workspace and target dataset can run at a time; a second concurrent
          request is rejected.

        ## Request Parameters
        - **channel_ids** (List[str]): Slack channel IDs (C... or G...), not channel names
          or DMs; duplicates are dropped. Channel selections also require days or oldest,
          and at least one channel, thread link or thread must be given. At most 50 items.
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        - **days** (Optional[int]): Length of the window ending at the time of the import,
          as an alternative to oldest and latest; giving both is rejected. With no bound at
          all the window covers the last 7 days. Must be between 1 and 3650.
        - **latest** (Optional[datetime]): End of the date window; must carry a timezone,
          come after oldest and be no later than now, and cannot be combined with days.
          Defaults to the time of the import.
        - **max_messages** (int): Cap on messages fetched for this import; reaching it fails
          the import instead of returning a truncated snapshot. Must be between 1 and
          500000. Defaults to 50000.
        - **max_requests** (int): Cap on Slack API calls made for this import; reaching it
          fails the import instead of returning a partial snapshot. Must be between 1 and
          10000. Defaults to 1000.
        - **oldest** (Optional[datetime]): Start of the date window; must carry a timezone,
          precede latest, and cannot be combined with days.
        - **thread_links** (List[str]): Slack message permalinks of the form
          https://<workspace>.slack.com/archives/C.../p...; duplicates are dropped, other
          URLs are rejected, and each link selects a whole conversation regardless of the
          date window. At most 50 items.
        - **thread_mode** (Literal['started', 'active']): 'started' selects threads whose
          root message falls in the window; 'active' also scans older roots and selects
          threads with any message in the window. Neither acts as a retention policy. One
          of: 'started', 'active'. Defaults to 'started'.
        - **threads** (List[SlackThread]): Threads named by channel id and root timestamp;
          each selects the entire conversation with all available replies, independent of
          the date window. At most 50 items.
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
        """Configure or disable scheduled Slack history sync for a workspace.

        Returns 400 when the selection is missing or invalid or when someone other than the
        connection owner disables the sync, and 403 when the caller lacks read, write and
        delete access to the target dataset. The saved settings replace a relative days
        window with the resolved oldest date, and the response echoes what was stored.

        ## Path Parameters
        - **team_id** (str): Slack workspace whose sync configuration is written. Enabling
          checks access to the selection's dataset; disabling is allowed only for the owner
          of the Slack connection.

        ## Request Parameters
        - **enabled** (bool): Turns the scheduled re-import on or off. Enabling requires a
          selection plus read, write and delete access to its dataset; disabling only
          requires owning the Slack connection. Defaults to False.
        - **interval_seconds** (int): How often the sync worker re-runs the stored
          selection. Must be between 600 and 604800. Defaults to 21600.
        - **selection** (Optional[SlackHistoryRequest]): Import selection the sync repeats;
          it also identifies the dataset being configured, so it is required even when
          disabling. A relative days window is resolved once into a fixed oldest date, and
          latest must not be set.
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
        """Report stored sync settings and recent import reports for a workspace.

        Returns 404 when the workspace has no active Slack connection owned by the caller.
        The body holds sync (per-dataset sync settings), reports (per-dataset import reports
        with status, timestamps and result) and worker_enabled, which says whether the
        background sync worker is configured on this server.

        ## Path Parameters
        - **team_id** (str): Slack workspace to report on; a workspace with no active
          connection owned by the caller answers 404.
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
