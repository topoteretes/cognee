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
        - **max_messages** (int): Upper limit on how many messages this import fetches.
          Defaults to 50000.
        - **max_requests** (int): Upper limit on how many Slack API requests this import
          issues. Defaults to 1000.
        - **oldest** (Optional[datetime]): Lower time bound of the messages fetched.
        - **thread_links** (List[str]): Slack permalinks of individual threads to include in
          the import.
        - **thread_mode** (Literal['started', 'active']): One of: 'started', 'active'. Defaults to
          'started'.
        - **threads** (List[SlackThread]): Channel and thread timestamp pairs identifying
          individual threads to import.
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
        """Set sync — PUT /api/v1/slack/history/{team_id}/sync.

        ## Path Parameters
        - **team_id** (str): Slack workspace whose recurring history sync is configured.

        ## Request Parameters
        - **enabled** (bool): Turns the recurring history sync on or off for this workspace.
          Defaults to False.
        - **interval_seconds** (int): Seconds to wait between recurring sync runs. Defaults
          to 21600.
        - **selection** (Optional[SlackHistoryRequest]): Channel, thread, and time-window
          selection that each recurring sync run re-imports.
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
        """History status — GET /api/v1/slack/history/{team_id}.

        ## Path Parameters
        - **team_id** (str): Slack workspace whose sync settings and last import reports are
          returned.
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
