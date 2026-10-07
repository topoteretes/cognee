"""Sync a Linear workspace into cognee memory through the SDK's DLT source.

The agent integration reads from whatever its dataset holds, so this fills one
dataset per workspace (``linear_<url_key>``) with one DLT table per team: the
team's issues with their comments folded in, and its projects. The bundled
``linear_source`` owns paging, cursors and rendering; this module owns which
teams, the token, and what happens to documents the old text path wrote.

* An install or a manual sync walks every selected team (all teams the app was
  granted when nothing is selected).
* A webhook syncs only the teams it names, and only for a connection that has
  finished a full pass: an upgraded install never starts a backfill on its own.
  A webhook that arrives while a sync runs is queued, not dropped.
* After a pass in which no team failed, deselected tables are retired and the
  plain-text issue documents of the former path are forgotten.

Syncs run with ``self_improvement=False`` (see ``ingestion.sync_scopes``):
``improve()`` is a whole-graph enrichment pass with LLM cost attached, far too
heavy to fire on every delivery.
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.linear.client import graphql
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.shared.logging_utils import get_logger

logger = get_logger("linear_sync")

LINEAR_DATASET_PREFIX = "linear"
PROVIDER = "linear"

# provider_metadata keys. Selection is three-state like Drive's and Gmail's:
# absent/None means every team the app was granted, [] none, a list those teams.
SELECTION_KEY = "selected_team_ids"
# Set once a full pass has gone through without a hard failure. Webhooks sync
# nothing before it, so an upgraded install never starts a backfill on its own.
SEEDED_KEY = "dlt_seeded"
# Set once the former text path's documents have been forgotten.
LEGACY_CLEANED_KEY = "legacy_cleaned"
# ISO time of the full pass that was cut short by the rate limit or the request
# budget, None when the last full pass was not. Only a full pass writes it:
# ``last_sync_counts`` is replaced by every run, a webhook's partial one included.
RESUME_KEY = "resume_needed_at"
RESUMABLE_COUNT_KEYS = ("failed_rate_limit", "failed_budget")

SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"

_TEAMS_PAGE_SIZE = 100

_TEAMS_QUERY = """
query LinearTeams($first: Int!, $after: String) {
  teams(first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { id key name description visibility }
  }
}
"""

# What the former path wrote, as plain text with no source tag:
# ``Linear issue <ID>: <title>`` (right-stripped, so a missing title leaves no
# space), ``URL: ...``, ``State: ...`` and optionally ``Description: ...``.
# Matched whole, so a note that merely starts with the same words survives.
_LEGACY_TEXT = re.compile(
    rb"\ALinear issue \S+:(?: [^\n]*)?\nURL: [^\n]*\nState: [^\n]*(?:\nDescription: |\Z)"
)

# dlt wraps the source's error, then remember() wraps that again.
_ERROR_CHAIN_DEPTH = 5

_EVENT_TYPES = ("Issue", "Comment", "Project")
_EVENT_ACTIONS = ("create", "update", "remove")


def dataset_name_for_org(url_key: str) -> str:
    """The one dataset a workspace's issues land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", url_key).strip("_").lower()
    return f"{LINEAR_DATASET_PREFIX}_{slug or 'workspace'}"


def dataset_name_for_credential(credential: IntegrationCredential) -> str:
    url_key = (credential.provider_metadata or {}).get("organization_url_key") or str(
        credential.provider_account_id
    )
    return dataset_name_for_org(url_key)


async def list_teams(credential: IntegrationCredential) -> list[dict[str, Any]]:
    """The teams the app token can access."""
    # Imported here: the adapter imports this module to wire its hooks.
    from cognee.modules.integrations.linear.adapter import call_with_token

    teams: dict[str, dict[str, Any]] = {}
    after = None
    while True:
        data = await call_with_token(
            credential,
            lambda token, after=after: graphql(
                token, _TEAMS_QUERY, {"first": _TEAMS_PAGE_SIZE, "after": after}
            ),
        )
        connection = data["teams"]
        for node in connection["nodes"]:
            teams.setdefault(node["id"], node)
        if not connection["pageInfo"]["hasNextPage"]:
            return list(teams.values())
        after = connection["pageInfo"]["endCursor"]


class _LinearService:
    """The DLT worker thread's Linear client, with its token taken from the API loop.

    The source never refreshes a token. This takes one per request from
    ``access_token_for`` (which refreshes an expiring one) and, when Linear
    answers 401, asks for a replacement once and retries.
    """

    def __init__(self, credential: IntegrationCredential, loop: asyncio.AbstractEventLoop):
        self._credential = credential
        self._loop = loop
        self.rate_limit: dict[str, int | None] = {}

    def _token(self, rejected: str | None = None) -> str:
        from cognee.modules.integrations.linear.adapter import access_token_for

        future = asyncio.run_coroutine_threadsafe(
            access_token_for(self._credential, rejected=rejected), self._loop
        )
        return future.result()

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        from cognee.tasks.ingestion.connectors.linear import LinearAuthError, LinearClient

        token = self._token()
        client = LinearClient(token)
        try:
            try:
                return client.execute(query, variables)
            except LinearAuthError:
                fresh = self._token(rejected=token)
                if fresh == token:
                    raise
                client = LinearClient(fresh)
                return client.execute(query, variables)
        finally:
            self.rate_limit = client.rate_limit


def _classify_error(error: BaseException) -> str:
    """The count key a failed team is reported under, from the source's typed errors."""
    from cognee.tasks.ingestion.connectors.linear import LinearAuthError, LinearTeamNotFoundError

    # dlt and ingestion each wrap the source's error once, so the typed error is a
    # few links down: one step per link, and no further than that.
    current: BaseException | None = error
    for _ in range(_ERROR_CHAIN_DEPTH):
        if current is None:
            break
        if isinstance(current, LinearTeamNotFoundError):
            return "failed_team_not_found"
        if isinstance(current, LinearAuthError):
            return "failed_auth"
        current = current.__cause__ or current.__context__
    return "failed_ingestion"


def team_ids_from_event(payload: dict[str, Any]) -> list[str]:
    """The teams a webhook delivery names, whichever entity it is about."""
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        return []
    event_type = payload.get("type")
    if event_type == "Issue":
        candidates = [data.get("teamId")]
    elif event_type == "Comment":
        issue = data.get("issue")
        candidates = [issue.get("teamId") if isinstance(issue, dict) else None, data.get("teamId")]
    elif event_type == "Project":
        team_ids = data.get("teamIds")
        candidates = list(team_ids) if isinstance(team_ids, list) else []
    else:
        return []
    return list(dict.fromkeys(c for c in candidates if isinstance(c, str) and c))


def is_team_event(payload: dict[str, Any]) -> bool:
    return payload.get("type") in _EVENT_TYPES and payload.get("action") in _EVENT_ACTIONS


async def forget_legacy_documents(credential: IntegrationCredential, dataset_name: str) -> int:
    """Forget the plain-text issue documents the former path wrote to this dataset.

    Only plain-text rows the owner added, in the owner's dataset, with no source
    tag whose text has the former path's exact shape. A document a user
    remembered into the dataset by hand, or one a collaborator added, is left
    alone. Returns how many went.
    """
    from sqlalchemy import select

    from cognee.api.v1.forget.forget import forget
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.files.utils.open_data_file import open_data_file
    from cognee.modules.data.models import Data, Dataset
    from cognee.modules.users.methods import get_user

    async with get_relational_engine().get_async_session() as db:
        result = await db.execute(
            select(Data.id, Data.dataset_id, Data.raw_data_location, Data.system_metadata)
            .join(Dataset, Data.dataset_id == Dataset.id)
            .where(
                Dataset.name == dataset_name,
                Dataset.owner_id == credential.user_id,
                Data.owner_id == credential.user_id,
                Data.extension == "txt",
            )
        )
        untagged = [row for row in result.all() if not row.system_metadata]

    legacy = []
    for row in untagged:
        try:
            async with open_data_file(row.raw_data_location, mode="rb") as file:
                text = file.read()
        except (OSError, ValueError):
            # Unreadable means skipped, never deleted.
            logger.warning("Could not read document %s while looking for old Linear text", row.id)
            continue
        if _LEGACY_TEXT.match(text):
            legacy.append(row)
    if not legacy:
        return 0

    owner = await get_user(credential.user_id)
    forgotten = 0
    for row in legacy:
        await ingestion.require_active_credential(credential)
        await forget(data_id=row.id, dataset_id=row.dataset_id, user=owner)
        forgotten += 1
    logger.info(
        "Forgot %d plain-text Linear issue documents in dataset %s", forgotten, dataset_name
    )
    return forgotten


async def sync_linear(credential: IntegrationCredential, team_ids: list[str] | None = None) -> None:
    """Run one sync and record its outcome. ``team_ids`` limits it to those teams."""

    async def sync_source(
        current: IntegrationCredential, counts: dict[str, int]
    ) -> tuple[str, dict[str, int]]:
        try:
            return await _sync_source(current, counts, team_ids)
        except CredentialInactiveError:
            raise
        except Exception:
            # A first full pass that fails outright (the team listing hit the rate
            # limit, a 5xx, the network) leaves no marker, so nothing would run it
            # again. Stamp it for the resume worker. Once a connection is seeded a
            # failure is not retried this way: a permanent one would loop.
            if team_ids is None and not (current.provider_metadata or {}).get(SEEDED_KEY):
                try:
                    await _mark(current, {RESUME_KEY: datetime.now(timezone.utc).isoformat()})
                except Exception:  # never replace the error in flight
                    logger.exception("Could not record that the Linear sync needs a retry")
            raise

    await ingestion.run_sync(PROVIDER, credential, sync_source)


# Teams a webhook named while a sync was running, by account. In-process like
# ``ingestion._running_syncs``.
_pending_teams: dict[str, set[str]] = {}


async def request_sync(
    credential: IntegrationCredential, team_ids: list[str] | None = None
) -> bool:
    """Sync now, or queue the named teams behind the sync that is running.

    ``run_sync`` drops a call while one runs. A webhook's change would then wait
    for the next trigger, so its teams are queued and synced as soon as the
    running sync ends. A full sync (``team_ids=None``) that finds one running is
    dropped, as it is for Drive and Gmail. Returns whether the request ran or
    was queued; teams that a failing run had taken are queued again.
    """
    account = str(credential.provider_account_id)
    if ingestion.sync_is_running(PROVIDER, account):
        if not team_ids:
            return False
        _pending_teams.setdefault(account, set()).update(team_ids)
        return True
    # Nothing awaits between the check above and ``run_sync`` marking the account
    # as running, so two callers cannot both pass it.
    queued = _pending_teams.pop(account, set())
    wanted: set[str] | None = None if team_ids is None else set(team_ids) | queued
    while True:
        try:
            await sync_linear(credential, None if wanted is None else sorted(wanted))
        except CredentialInactiveError:
            _pending_teams.pop(account, None)
            raise
        except Exception:
            if wanted:
                _pending_teams.setdefault(account, set()).update(wanted)
            raise
        wanted = _pending_teams.pop(account, set())
        if not wanted:
            return True


async def _mark(credential: IntegrationCredential, patch: dict[str, Any]) -> None:
    """Write sync bookkeeping to the connection, only while it is still active.

    A sync outlives the install that started it, and the connection's metadata
    survives a reconnect: a late write would give the next install this one's
    markers. What is already stored is compared, not the snapshot the run started
    with: a reinstall can have reset a marker the snapshot still shows as set.
    """
    from cognee.modules.integrations.credentials import update_provider_metadata

    current = await ingestion.require_active_credential(credential)
    stored = (current.provider_metadata or {}) if current is not None else {}
    patch = {key: value for key, value in patch.items() if stored.get(key) != value}
    if not patch:
        return
    await update_provider_metadata(PROVIDER, str(credential.provider_account_id), patch)


async def _retire(credential: IntegrationCredential, dataset: str, retained: set[str]) -> None:
    """Retire deselected tables, unless another workspace of this user shares the dataset.

    The dataset name is a lossy slug of the workspace's url key and, unlike Drive
    and Gmail, carries no account suffix. ``retire_resources`` empties every
    table in the dataset that is not in ``retained``, so with two workspaces
    that normalise to one name one would empty the other's.
    """
    if await _dataset_is_shared(credential, dataset):
        logger.warning(
            "Not retiring Linear tables for organization %s: another workspace of this user "
            "uses dataset %s too",
            credential.provider_account_id,
            dataset,
        )
        return
    await ingestion.retire_resources(PROVIDER, credential, dataset, retained)


async def _dataset_is_shared(credential: IntegrationCredential, dataset: str) -> bool:
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.integrations.models.IntegrationCredential import (
        IntegrationCredential as CredentialRow,
    )

    async with get_relational_engine().get_async_session() as db:
        others = (
            (
                await db.execute(
                    select(CredentialRow).where(
                        CredentialRow.provider == PROVIDER,
                        CredentialRow.status == "active",
                        CredentialRow.user_id == credential.user_id,
                        CredentialRow.provider_account_id != credential.provider_account_id,
                    )
                )
            )
            .scalars()
            .all()
        )
    return any(dataset_name_for_credential(other) == dataset for other in others)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int], team_ids: list[str] | None
) -> tuple[str, dict[str, int]]:
    metadata = credential.provider_metadata or {}
    selected = metadata.get(SELECTION_KEY)
    if selected is not None and (
        not isinstance(selected, list) or not all(isinstance(t, str) and t for t in selected)
    ):
        raise ValueError("Linear selection must be a list of team IDs or null")
    dataset = dataset_name_for_credential(credential)
    partial = team_ids is not None

    if selected == []:
        if not partial:
            await _retire(credential, dataset, set())
            # A full pass with nothing selected is a complete pass: once the user
            # selects teams, webhooks for them can seed them.
            await _mark(credential, {SEEDED_KEY: True, RESUME_KEY: None})
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}

    if partial:
        wanted = set(team_ids or [])
        if selected is not None:
            scopes = [t for t in selected if t in wanted]
        else:
            # A delivery can name a team the app was never granted, or one that is
            # gone: only granted teams are synced, so it cannot fail the connection.
            granted = {str(team["id"]) for team in await list_teams(credential)}
            scopes = [t for t in (team_ids or []) if t in granted]
    elif selected is not None:
        scopes = list(dict.fromkeys(selected))
    else:
        scopes = [str(team["id"]) for team in await list_teams(credential)]
    scopes = list(dict.fromkeys(scopes))
    if not scopes:
        return SYNC_STATUS_OK, counts

    service = _LinearService(credential, asyncio.get_running_loop())
    source_factory = ingestion.source_factory(PROVIDER)

    def make_source(team_id: str, resource_name: str, check_active):
        return source_factory(
            team_id=team_id,
            resource_name=resource_name,
            check_active=check_active,
            service=service,
        )

    retained = await ingestion.sync_scopes(
        PROVIDER,
        credential,
        counts,
        scopes=scopes,
        dataset_name=dataset,
        make_source=make_source,
        classify_error=_classify_error,
    )
    # A selected team that is gone fails on every run. It is reported, but it
    # must not keep the seed marker, the cleanup and the other teams' retirement
    # from ever happening.
    hard_failures = counts["failed"] - counts.get("failed_team_not_found", 0)
    if not partial:
        cut_short = any(counts.get(key) for key in RESUMABLE_COUNT_KEYS)
        resume_at = datetime.now(timezone.utc).isoformat() if cut_short else None
    if hard_failures > 0:
        if not partial:
            await _mark(credential, {RESUME_KEY: resume_at})
        return SYNC_STATUS_DEGRADED, counts

    if not partial:
        # First, so a failing retire or cleanup below cannot keep webhooks off.
        await _mark(credential, {SEEDED_KEY: True, RESUME_KEY: None})
        # Retire only when the user chose teams: with no selection every granted
        # team is synced and a team missing from one listing must not be deleted.
        if selected is not None:
            await _retire(credential, dataset, retained)
        if not metadata.get(LEGACY_CLEANED_KEY):
            try:
                await forget_legacy_documents(credential, dataset)
                await _mark(credential, {LEGACY_CLEANED_KEY: True})
            except CredentialInactiveError:
                raise
            except Exception:
                logger.exception(
                    "Forgetting the old Linear text documents of organization %s failed; "
                    "it is retried on the next full sync",
                    credential.provider_account_id,
                )
    return (SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK), counts
