"""Validated GitHub sync policy and explicit scheduler entry point.

Scheduling never starts a background service. Call ``run_due_github_syncs`` from
an external scheduler. Local locks serialize a single SDK worker's queued runs;
run one scheduler/worker per installation, as with the SDK's dataset locks.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

import aiohttp
from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from cognee.api.DTO import InDTO
from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks.session_lock import session_lock
from cognee.modules.integrations.credentials import STATUS_ACTIVE, CredentialInactiveError
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

SETTINGS_KEY = "github_sync_settings"
NEXT_SYNC_KEY = "github_next_sync_at"

# What a sync of one account can fail with: a cognee error, a provider, network or I/O failure,
# or the credential store. Anything else is a bug and stops the run. Messages are never logged,
# since they can carry credentials.
_ACCOUNT_FAILURES = (
    CogneeApiError,
    OSError,
    RuntimeError,
    ValueError,
    SQLAlchemyError,
    aiohttp.ClientError,
    asyncio.TimeoutError,
)


class SyncSchedule(InDTO):
    """Intervals anchored to local midnight or the selected window start."""

    model_config = ConfigDict(extra="forbid")
    interval_minutes: StrictInt = Field(gt=0)
    weekdays: list[StrictInt] = Field(default_factory=lambda: list(range(7)))
    timezone: str = "UTC"
    start_time: str | None = None
    end_time: str | None = None

    @field_validator("timezone")
    @classmethod
    def valid_zone(cls, value):
        try:
            ZoneInfo(value)
        except (KeyError, ValueError) as error:
            raise ValueError("timezone must be an IANA timezone") from error
        return value

    @field_validator("weekdays")
    @classmethod
    def valid_days(cls, value):
        if not value or any(day < 0 or day > 6 for day in value):
            raise ValueError("weekdays must contain days from 0 (Monday) to 6 (Sunday)")
        return sorted(set(value))

    @field_validator("start_time", "end_time")
    @classmethod
    def valid_time(cls, value):
        if value is not None:
            try:
                parsed = time.fromisoformat(value)
            except ValueError as error:
                raise ValueError("time must use HH:MM") from error
            if len(value) != 5 or parsed.tzinfo is not None:
                raise ValueError("time must use HH:MM")
        return value

    @model_validator(mode="after")
    def valid_window(self):
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if self.start_time is not None and self.start_time >= self.end_time:
            raise ValueError("end_time must be after start_time on the same day")
        return self


class GithubRepositorySettings(InDTO):
    """One stable GitHub repository id's branch and file selection."""

    model_config = ConfigDict(extra="forbid")
    branch: str | None = None
    include_paths: list[str] | None = None
    exclude_paths: list[str] = Field(default_factory=list)
    include_documents: StrictBool = False

    @field_validator("include_documents")
    @classmethod
    def documents_unavailable(cls, value):
        if value:
            raise ValueError(
                "GitHub document sync is unavailable until source-owned document removal is supported"
            )
        return value

    @field_validator("branch")
    @classmethod
    def valid_branch(cls, value):
        if value is not None:
            from cognee.tasks.code_graph.resolve_repo import check_branch_name

            check_branch_name(value)
        return value

    @model_validator(mode="after")
    def valid_patterns(self):
        from cognee.exceptions import CogneeValidationError
        from cognee.tasks.code_graph.repo_selection import validate_path_patterns

        try:
            validate_path_patterns(self.include_paths, "include_paths", optional=True)
            validate_path_patterns(self.exclude_paths, "exclude_paths")
        except CogneeValidationError as error:
            raise ValueError(str(error)) from error
        return self


class GithubSyncSettings(InDTO):
    """None repositories means all; an empty mapping explicitly selects none."""

    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool = True
    mode: Literal["push", "scheduled", "manual"] = "push"
    schedule: SyncSchedule | None = None
    repositories: dict[str, GithubRepositorySettings] | None = None
    first_sync: Literal["immediate", "scheduled"] = "immediate"

    @model_validator(mode="after")
    def valid_settings(self):
        if self.mode == "scheduled" and self.schedule is None:
            raise ValueError("scheduled mode requires a schedule")
        if self.first_sync == "scheduled" and self.mode != "scheduled":
            raise ValueError("first_sync='scheduled' requires scheduled mode")
        for key in self.repositories or {}:
            if not key.isascii() or not key.isdigit() or int(key) <= 0 or str(int(key)) != key:
                raise ValueError("repository keys must be positive numeric GitHub ids")
        return self


def read_sync_settings(credential: IntegrationCredential) -> GithubSyncSettings:
    return GithubSyncSettings.model_validate(
        (credential.provider_metadata or {}).get(SETTINGS_KEY, {})
    )


def next_sync_at(
    settings: GithubSyncSettings, after: datetime, *, inclusive: bool = False
) -> datetime | None:
    """Next UTC slot; nonexistent DST wall times are skipped, both folds run."""
    if after.tzinfo is None:
        raise ValueError("after must be timezone-aware")
    if not settings.enabled or settings.mode != "scheduled":
        return None
    schedule = settings.schedule
    zone = ZoneInfo(schedule.timezone)
    day = after.astimezone(zone).date()
    start = time.fromisoformat(schedule.start_time or "00:00")
    end = time.fromisoformat(schedule.end_time) if schedule.end_time else None
    while True:
        if day.weekday() in schedule.weekdays:
            wall = datetime.combine(day, start)
            stop = (
                datetime.combine(day, end)
                if end
                else datetime.combine(day + timedelta(days=1), time())
            )
            candidates = []
            while wall < stop:
                for fold in (0, 1):
                    candidate = wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
                    if candidate.astimezone(zone).replace(tzinfo=None) != wall:
                        continue
                    if candidate > after or (inclusive and candidate == after):
                        candidates.append(candidate)
                # An interval beyond the current day's window has no more slots.
                remaining = (stop - wall).total_seconds() / 60
                if schedule.interval_minutes >= remaining:
                    break
                wall += timedelta(minutes=schedule.interval_minutes)
            if candidates:
                return min(candidates)
        day += timedelta(days=1)


def selects_push(settings: GithubSyncSettings, repository: dict, branch: str) -> bool:
    """Whether the saved policy syncs a push of ``branch`` to this repository (a push payload).

    Uses only the payload and the saved policy: the branch a repository syncs is its policy
    branch, else the default branch GitHub reports in the payload.
    """
    policy = GithubRepositorySettings()
    if settings.repositories is not None:
        selected = settings.repositories.get(str(repository.get("id")))
        if selected is None:
            return False
        policy = selected
    return (policy.branch or repository.get("default_branch")) == branch


def within_schedule(settings: GithubSyncSettings, now: datetime) -> bool:
    """Whether an overdue sync may execute at this local day and time."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not settings.enabled or settings.mode != "scheduled":
        return False
    schedule = settings.schedule
    local = now.astimezone(ZoneInfo(schedule.timezone))
    if local.weekday() not in schedule.weekdays:
        return False
    return schedule.start_time is None or (
        time.fromisoformat(schedule.start_time)
        <= local.time()
        < time.fromisoformat(schedule.end_time)
    )


async def save_sync_settings(
    credential: IntegrationCredential,
    settings: GithubSyncSettings,
    *,
    acknowledge_removal: bool = False,
) -> IntegrationCredential:
    """Save policy; changing an existing content selection requires acknowledgement."""
    settings = GithubSyncSettings.model_validate(settings)
    async with session_lock(str(credential.id), "github_settings"):
        engine = get_relational_engine()
        async with engine.get_async_session() as db:
            result = await db.execute(
                select(IntegrationCredential)
                .where(
                    IntegrationCredential.id == credential.id,
                    IntegrationCredential.user_id == credential.user_id,
                    IntegrationCredential.workspace_id == credential.workspace_id,
                    IntegrationCredential.provider_account_id == credential.provider_account_id,
                    IntegrationCredential.status == STATUS_ACTIVE,
                )
                .with_for_update()
            )
            current = result.scalar_one_or_none()
            if current is None or current.provider != "github":
                raise CredentialInactiveError("GitHub connection is no longer active")
            old = read_sync_settings(current)
            if old.repositories != settings.repositories and not acknowledge_removal:
                raise ValueError("Changing repository selection requires acknowledge_removal=true")
            if old.repositories != settings.repositories:
                from cognee.modules.integrations.github.retire import assert_tracked_selection

                await assert_tracked_selection(db, current)
            metadata = dict(current.provider_metadata or {})
            metadata[SETTINGS_KEY] = settings.model_dump(mode="json")
            deadline = next_sync_at(settings, datetime.now(timezone.utc))
            metadata[NEXT_SYNC_KEY] = deadline.isoformat() if deadline else None
            current.provider_metadata = metadata
            await db.commit()
            await db.refresh(current)
            return current


async def run_due_github_syncs(
    now: datetime | None = None, *, user_id: UUID | None = None
) -> dict[str, str]:
    """Run due connections once; overdue slots coalesce into one full sync."""
    from cognee.modules.integrations.github.sync import sync_repositories

    if now is not None and now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    conditions = [
        IntegrationCredential.provider == "github",
        IntegrationCredential.status == STATUS_ACTIVE,
    ]
    if user_id is not None:
        conditions.append(IntegrationCredential.user_id == user_id)
    async with get_relational_engine().get_async_session() as db:
        result = await db.execute(select(IntegrationCredential).where(*conditions))
        credentials = list(result.scalars().all())
    outcomes = {}
    for credential in credentials:
        try:
            outcomes[str(credential.provider_account_id)] = await sync_repositories(
                credential, trigger="scheduled", now=now
            )
        except _ACCOUNT_FAILURES as error:  # one account's failure must not stop the others
            logger.error(
                "Scheduled GitHub sync for account %s failed: %s",
                credential.provider_account_id,
                type(error).__name__,
            )
            outcomes[str(credential.provider_account_id)] = "failed"
    return outcomes


async def claim_scheduled_sync(
    credential: IntegrationCredential, now: datetime
) -> tuple[IntegrationCredential, str]:
    """Advance one due slot under the same guarded row lock used by settings saves."""
    async with (
        session_lock(str(credential.id), "github_settings"),
        get_relational_engine().get_async_session() as db,
    ):
        result = await db.execute(
            select(IntegrationCredential)
            .where(
                IntegrationCredential.id == credential.id,
                IntegrationCredential.user_id == credential.user_id,
                IntegrationCredential.workspace_id == credential.workspace_id,
                IntegrationCredential.provider_account_id == credential.provider_account_id,
                IntegrationCredential.status == STATUS_ACTIVE,
            )
            .with_for_update()
        )
        current = result.scalar_one_or_none()
        if current is None:
            raise CredentialInactiveError("GitHub connection is no longer active")
        settings = read_sync_settings(current)
        if not settings.enabled or settings.mode != "scheduled":
            return current, "mode_skipped"
        if not within_schedule(settings, now):
            return current, "outside_schedule"
        metadata = dict(current.provider_metadata or {})
        raw = metadata.get(NEXT_SYNC_KEY)
        deadline = (
            datetime.fromisoformat(raw) if raw else next_sync_at(settings, now, inclusive=True)
        )
        if deadline > now:
            return current, "not_due"
        metadata[NEXT_SYNC_KEY] = next_sync_at(settings, now).isoformat()
        current.provider_metadata = metadata
        await db.commit()
        await db.refresh(current)
        return current, "claimed"
