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
        - **team_id** (str): Slack workspace whose connected account is used to fetch the
          selected history.

        ## Request Parameters
        - **channel_ids** (List[str]): Slack channel ids whose messages are fetched and
          indexed.
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        - **days** (Optional[int]): Relative time window, in days back from now, bounding
          the messages fetched.
        - **latest** (Optional[datetime]): Upper time bound of the messages fetched.
        - **max_messages** (int): Most messages the import may fetch, 1 to 500000. Reaching
          it fails the import; it never returns a truncated snapshot. Defaults to 50000.
        - **max_requests** (int): Most Slack API requests the import may make, 1 to 10000.
          Reaching it fails the import; it never returns a truncated snapshot. Defaults
          to 1000.
        - **oldest** (Optional[datetime]): Lower time bound of the messages fetched.
        - **thread_links** (List[str]): Slack permalinks of individual threads to include in
          the import.
        - **thread_mode** (Literal['started', 'active']): 'started' selects threads whose
          root message is in the window; 'active' also selects older threads with a message
          in the window. Defaults to 'started'.
        - **threads** (List[SlackThread]): Channel and thread timestamp pairs identifying
          individual threads to import.

        ## Selection Rules
        Breaking any of these is rejected with 422, except the last, which is a 400.
        - Select at least one of channel_ids, thread_links or threads. Each list holds
          at most 50 items.
        - channel_ids must be channel IDs (C... or G...), not channel names or DMs.
        - Channel imports need days or oldest. Thread links and threads are imported
          whole, regardless of the date window.
        - Use either days (1 to 3650) or oldest/latest, not both.
        - oldest and latest must include a timezone, and oldest must come before latest.
        - The window must end no later than now (400).

        ## Error Codes
        - **400 Bad Request**: The window ends in the future, or an import for this
          workspace and dataset is already running
        - **403 Forbidden**: Caller lacks read, write and delete access to the dataset
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
        """Turn a recurring history sync into one dataset on or off.

        Settings are stored per workspace and destination dataset (the selection's
        dataset_id). A days window is converted once into a fixed oldest date, so later
        runs keep the same start and pick up late replies. Returns the saved settings.

        ## Path Parameters
        - **team_id** (str): Slack workspace whose recurring history sync is configured.

        ## Request Parameters
        - **enabled** (bool): Turns the recurring history sync on or off for the selection's
          dataset. Defaults to False.
        - **interval_seconds** (int): Seconds between recurring sync runs, 600 to 604800.
          Defaults to 21600.
        - **selection** (Optional[SlackHistoryRequest]): Channel, thread and time-window
          selection that each run re-imports, with the same rules as POST
          /history/{team_id}/import. Required, including to disable a sync, because its
          dataset_id says which sync to change. It cannot set latest.

        ## Error Codes
        - **400 Bad Request**: No selection was sent, or a disabling caller does not own
          the Slack connection
        - **403 Forbidden**: Enabling caller lacks read, write and delete access to the
          dataset
        - **422 Unprocessable Entity**: enabled is true without a selection, the selection
          sets latest, or the selection breaks the import rules
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
        """Show a workspace's sync settings and latest import reports.

        Only the user who connected the workspace can read it; anyone else, or an
        inactive connection, gets a 404.

        ## Path Parameters
        - **team_id** (str): Slack workspace whose sync settings and last import reports are
          returned.

        ## Response
        - **sync**: Saved sync settings, keyed by destination dataset id.
        - **reports**: Latest import report per dataset id: status (running, completed or
          failed), started_at, finished_at, and the result counts or an error message.
        - **worker_enabled**: Whether this server runs the recurring sync worker. When it
          is false, saved syncs do not run.
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
