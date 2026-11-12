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
        - **team_id** (str): Slack workspace (team) id of the connected installation to read
          history from. Only the owner of an active connection for it may import. An import
          already running in this process for this workspace and the same dataset is
          rejected with 400.

        ## Request Parameters
        - **channel_ids** (List[str]): Slack channel ids (C… or G…) to import; names and DMs
          are rejected and duplicates dropped. Channel imports also require days or an
          oldest date, and at least one channel or thread must be selected. At most 50
          items.
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        - **days** (Optional[int]): Size of the window ending now, used instead of explicit
          dates; passing it together with oldest or latest is rejected. Must be between 1
          and 3650.
        - **latest** (Optional[datetime]): End of the date window; must be timezone-aware,
          later than oldest and no later than now, and cannot be combined with days.
          Omitted, the window ends at the time of the request.
        - **max_messages** (int): Cap on the messages fetched for this selection. Exceeding
          the cap fails the import with 400 rather than storing a truncated snapshot. Must be
          between 1 and 500000. Defaults to 50000.
        - **max_requests** (int): Cap on the Slack API requests this import may issue,
          retries included. Exceeding the cap fails the import with 400 rather than storing a
          truncated snapshot. Must be between 1 and 10000. Defaults to 1000.
        - **oldest** (Optional[datetime]): Start of the date window; must be timezone-aware
          and precede latest, and cannot be combined with days. Channel selections require
          it when days is not given.
        - **thread_links** (List[str]): Slack message permalinks
          (https://<workspace>.slack.com/archives/…) selecting whole conversations
          regardless of the date window; duplicates are dropped and links that cannot be
          parsed are rejected. At most 50 items.
        - **thread_mode** (Literal['started', 'active']): 'started' selects threads whose
          root message falls in the window; 'active' scans the channel's whole history up to
          latest and selects threads with any message in the window. Every page scanned
          counts toward max_messages and max_requests, so 'active' on a large channel can
          exceed them. Neither acts as a retention policy. One of: 'started', 'active'.
          Defaults to 'started'.
        - **threads** (List[SlackThread]): Threads named directly by channel id and root
          timestamp, as an alternative to thread_links; each selected thread is imported
          with all available replies. At most 50 items.
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
        """Configure or disable recurring Slack history sync for a dataset.

        Returns the stored settings, with the selection's days window resolved into a fixed
        oldest date so the boundary does not move between runs. An invalid or missing
        selection returns 400, whether enabling or disabling; enabling without read, write
        and delete access to the target dataset returns 403.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id whose sync settings are written.
          Configuring or disabling a sync is allowed only for the owner of the active Slack
          connection.

        ## Request Parameters
        - **enabled** (bool): When true the stored selection is refreshed on the interval
          and the caller is checked for access to its dataset; a selection is then required.
          Refreshes run only while the sync worker is enabled (worker_enabled on GET
          /api/v1/slack/history/{team_id}). When false the sync is turned off. Defaults to
          False.
        - **interval_seconds** (int): How often the stored selection is refreshed once sync
          is enabled; after a run that did not complete, the next attempt comes after 600
          seconds instead. Must be between 600 and 604800. Defaults to 21600.
        - **selection** (Optional[SlackHistoryRequest]): Import selection naming the dataset
          to configure or disable; required in both cases, and its latest date must be
          unset. Its days window is resolved once and stored as a fixed oldest date.
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

        Responds with sync and reports objects keyed by dataset id (empty objects when
        nothing has been configured or run) and worker_enabled, which is true only when
        history sync is enabled and a Slack client id is configured. Each report records
        status running, completed or failed, timestamps and, on success, the import result.
        A workspace with no active connection owned by the caller returns 404.

        ## Path Parameters
        - **team_id** (str): Slack workspace (team) id to report on. A workspace without an
          active connection owned by the caller returns 404.
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
