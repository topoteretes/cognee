"""``@cognee status``: who the member is to cognee, what cognee does with this channel, and
the workspace's sync, each row from data the code already has. Slack and the stores are
stood in for at the module's seams; a row that cannot be computed says "Unknown"."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.integrations.slack import channel_membership as membership
from cognee.modules.integrations.slack import status

TEAM = "T0TEAM"
INSTALLER = "U0INSTALLER"
OWNER = uuid4()
# Two minutes ago: both the next regular sync and a resume are still ahead.
LAST = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=2)


def _credential(**metadata):
    return SimpleNamespace(
        id=uuid4(),
        status="active",
        provider_account_id=TEAM,
        user_id=OWNER,
        provider_metadata={
            "installed_by_slack_user_id": INSTALLER,
            "selected_channels": ["C1", "C3", "G0SECRET"],
            "last_sync_counts": {"emitted": 12, "failed": 0},
        }
        | metadata,
        last_synced_at=LAST,
        sync_status="ok",
    )


@pytest.fixture
def stores(monkeypatch):
    """A linked member, two public channels of the three selected, 40 stored items, no
    sync running and no pause."""
    state = SimpleNamespace(
        public={"C1", "C3"},
        external=False,
        running=False,
        paused=False,
        items=40,
    )
    monkeypatch.setattr(
        status, "get_user", AsyncMock(return_value=SimpleNamespace(email="ana@acme.example"))
    )
    monkeypatch.setattr(
        status, "workspace_item_count", AsyncMock(side_effect=lambda *a: state.items)
    )
    monkeypatch.setattr(status, "sync_is_running", lambda *a: state.running)
    monkeypatch.setattr(status, "is_paused", AsyncMock(side_effect=lambda *a: state.paused))
    monkeypatch.setattr(
        membership, "public_channels", AsyncMock(side_effect=lambda *a: state.public)
    )
    monkeypatch.setattr(
        membership,
        "shared_with_other_organizations",
        AsyncMock(side_effect=lambda *a: state.external),
    )
    return state


def _fields(message: dict) -> dict[str, str]:
    """Each field's bold label to its value."""
    rows = {}
    for block in message["blocks"]:
        for field in block.get("fields", []):
            label, _, value = field["text"].partition("\n")
            rows[label.strip("*")] = value
    return rows


async def _status(credential, channel="C1", owner=OWNER):
    return await status.status_message(credential, owner, channel=channel)


@pytest.mark.asyncio
async def test_a_linked_member_in_a_synced_channel_sees_every_row(stores):
    message = await _status(_credential())
    rows = _fields(message)

    assert rows["You"] == "Linked as ana@acme.example"
    assert rows["Connected by"] == f"<@{INSTALLER}>"
    assert rows["This channel"] == "Synced · commands on"
    assert rows["Answers here"] == "Read <#C1> and 1 public channel"
    assert rows["Memory"] == "3 channels synced (2 public, 1 private) · 40 conversations"
    epoch = int(LAST.timestamp())
    assert rows["Last sync"].startswith(
        f"<!date^{epoch}^{{date_short_pretty}} at {{time}}|{LAST:%Y-%m-%d %H:%M} UTC>"
    )
    assert "· 12 new, 0 failed" in rows["Last sync"]
    nxt = int((LAST + timedelta(hours=6)).timestamp())
    assert rows["Next sync"].startswith(f"<!date^{nxt}^")
    (footer,) = [block for block in message["blocks"] if block["type"] == "context"]
    assert footer["elements"][0]["text"] == (
        "Manage channels in cognee → Integrations. Only you can see this."
    )
    assert "divider" in [block["type"] for block in message["blocks"]]
    assert "Linked as ana@acme.example" in message["text"]


@pytest.mark.asyncio
async def test_a_private_channel_is_only_ever_counted(stores):
    """The member may not be in a selected private channel: its id or name never shows."""
    message = await _status(_credential())

    assert "G0SECRET" not in str(message)
    assert "secret" not in str(message).lower()


@pytest.mark.asyncio
async def test_an_unlinked_member_is_told_how_to_link(stores):
    rows = _fields(await _status(_credential(), owner=None))

    assert rows["You"] == "Not linked: send `@cognee link`"


@pytest.mark.asyncio
async def test_a_channel_that_is_not_synced_names_who_can_add_it(stores):
    rows = _fields(await _status(_credential(), channel="C9"))

    assert rows["This channel"] == f"Not synced: ask <@{INSTALLER}> to add it · commands on"
    assert rows["Answers here"] == "Read 2 public channels"


@pytest.mark.asyncio
async def test_commands_switched_off_here_are_said(stores):
    rows = _fields(await _status(_credential(allowed_channel_ids=["C3"])))

    assert rows["This channel"] == "Synced · Commands are off here"


@pytest.mark.asyncio
async def test_a_slack_connect_channel_reads_itself_alone(stores):
    stores.external = True

    rows = _fields(await _status(_credential()))

    assert rows["Answers here"] == "Read this channel only (shared with another organization)"


@pytest.mark.asyncio
async def test_a_direct_message_says_so(stores):
    rows = _fields(await _status(_credential(), channel=None))

    assert rows["This channel"] == "This is a direct message"
    assert rows["Answers here"] == "Your channels and your memory"


@pytest.mark.asyncio
async def test_a_degraded_sync_says_why_in_the_sentences_the_integrations_page_uses(stores):
    credential = _credential(
        last_sync_counts={"emitted": 3, "failed": 2, "failed_budget": 1, "failed_auth": 1}
    )
    credential.sync_status = "degraded"

    rows = _fields(await _status(credential))

    assert "· 3 new, 2 failed" in rows["Last sync"]
    assert (
        "Large history: the sync continues where it stopped in about ten minutes."
        in (rows["Last sync"])
    )
    assert "Slack signed Cognee out. Reconnect to keep syncing." in rows["Last sync"]
    # A cut-short run resumes ten minutes after it, not six hours (``is_due``).
    nxt = int((LAST + timedelta(seconds=600)).timestamp())
    assert rows["Next sync"].startswith(f"<!date^{nxt}^")


@pytest.mark.asyncio
async def test_a_paused_connection_says_the_owner_paused_it(stores):
    stores.paused = True

    rows = _fields(await _status(_credential()))

    assert rows["Next sync"] == "Paused by the owner"


@pytest.mark.asyncio
async def test_a_running_sync_says_so(stores):
    stores.running = True

    rows = _fields(await _status(_credential()))

    assert rows["Next sync"] == "Syncing now"


@pytest.mark.asyncio
async def test_a_row_slack_cannot_answer_says_unknown_and_the_rest_still_show(stores):
    membership.public_channels.side_effect = RuntimeError("slack down")

    rows = _fields(await _status(_credential()))

    assert rows["Memory"] == "Unknown"
    assert rows["Answers here"] == "Unknown"
    assert rows["You"] == "Linked as ana@acme.example"


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["workspace_item_count", "is_paused"])
async def test_a_failing_read_of_cognees_own_data_fails_the_status(stores, read):
    """Igor (D6): a catch-all around a local read hides a bug as "Unknown"."""
    getattr(status, read).side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await _status(_credential())


@pytest.mark.asyncio
async def test_a_member_linked_to_a_deleted_user_sees_unknown(stores):
    from cognee.infrastructure.databases.exceptions import EntityNotFoundError

    status.get_user.side_effect = EntityNotFoundError(message="gone")

    assert _fields(await _status(_credential()))["You"] == "Unknown"


@pytest.mark.asyncio
async def test_one_channel_and_one_conversation_read_in_the_singular(stores):
    """Live, 2026-11-12: the status said "1 channels synced ... 4 conversations"."""
    stores.items = 1
    message = await _status(_credential(selected_channels=["C1"]))

    assert _fields(message)["Memory"].startswith("1 channel synced (")
    assert _fields(message)["Memory"].endswith(" · 1 conversation")
