"""Selection contracts for Slack history ingestion (no memory state)."""

import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

CHANNEL = re.compile(r"^[CG][A-Z0-9]+$")
TIMESTAMP = re.compile(r"^\d{10,}\.\d{6}$")


class SlackHistoryError(ValueError):
    """An actionable, safe-to-display connector error."""


def parse_thread_link(link: str) -> tuple[str, str]:
    """Parse a Slack permalink locally; never fetch a caller-supplied URL."""
    url = urlparse(link)
    if (
        url.scheme != "https"
        or not (url.hostname or "").endswith(".slack.com")
        or url.username
        or url.password
        or url.port not in (None, 443)
    ):
        raise ValueError("Use an https://<workspace>.slack.com/archives/... message link.")
    match = re.fullmatch(r"/archives/([CG][A-Z0-9]+)/p(\d{16,})/?", url.path)
    if not match:
        raise ValueError("Expected a Slack channel message or thread link.")
    channel, digits = match.groups()
    ts = (parse_qs(url.query).get("thread_ts") or [f"{digits[:-6]}.{digits[-6:]}"])[0]
    if not TIMESTAMP.fullmatch(ts):
        raise ValueError("Invalid Slack thread timestamp.")
    return channel, ts


class SlackThread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_id: str = Field(pattern=r"^[CG][A-Z0-9]+$")
    ts: str = Field(pattern=r"^\d{10,}\.\d{6}$")


class SlackHistoryRequest(BaseModel):
    """Date bounds select conversations; selected threads include all available replies.

    ``started`` selects roots in the window. ``active`` also scans older roots
    and selects threads with a message in the window. Neither is a retention
    policy. Explicit thread links select entire conversations independently
    of the date window. All bounds are timezone-aware.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID
    channel_ids: list[str] = Field(default_factory=list, max_length=50)
    thread_links: list[str] = Field(default_factory=list, max_length=50)
    threads: list[SlackThread] = Field(default_factory=list, max_length=50)
    days: int | None = Field(default=None, ge=1, le=3650)
    oldest: datetime | None = None
    latest: datetime | None = None
    thread_mode: Literal["started", "active"] = "started"
    # A cap is a failed import, never a successful truncated snapshot.
    max_requests: int = Field(default=1000, ge=1, le=10000)
    max_messages: int = Field(default=50000, ge=1, le=500000)

    @model_validator(mode="after")
    def validate_selection(self):
        if not self.channel_ids and not self.thread_links and not self.threads:
            raise ValueError("Select at least one channel or thread link.")
        if any(not CHANNEL.fullmatch(channel) for channel in self.channel_ids):
            raise ValueError("Use Slack channel IDs (C... or G...), not channel names or DMs.")
        self.channel_ids = list(dict.fromkeys(self.channel_ids))
        self.thread_links = list(dict.fromkeys(self.thread_links))
        for link in self.thread_links:
            parse_thread_link(link)
        if self.days is not None and (self.oldest is not None or self.latest is not None):
            raise ValueError("Use days or explicit oldest/latest dates, not both.")
        for bound in (self.oldest, self.latest):
            if bound is not None and (bound.tzinfo is None or bound.utcoffset() is None):
                raise ValueError("Dates must include a timezone, e.g. 2026-09-01T00:00:00Z.")
        if self.channel_ids and self.days is None and self.oldest is None:
            raise ValueError("Channel imports require days or an oldest date.")
        if self.oldest and self.latest and self.oldest >= self.latest:
            raise ValueError("oldest must precede latest.")
        return self

    def bounds(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        now = now or datetime.now(timezone.utc)
        oldest = self.oldest or (now - timedelta(days=self.days or 7))
        latest = self.latest or now
        if oldest >= latest or latest > now:
            raise SlackHistoryError("Choose a date range ending no later than now.")
        return oldest, latest


class SlackHistoryResult(BaseModel):
    dataset_id: UUID
    conversations: int = 0
    messages: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    status: Literal["completed"] = "completed"


class SlackSyncSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    selection: SlackHistoryRequest | None = None
    interval_seconds: int = Field(default=21600, ge=600, le=604800)

    @model_validator(mode="after")
    def validate_enabled(self):
        if self.enabled and self.selection is None:
            raise ValueError("An enabled sync needs a selection.")
        if self.selection and self.selection.latest:
            raise ValueError("Ongoing sync cannot have a fixed latest date.")
        return self
