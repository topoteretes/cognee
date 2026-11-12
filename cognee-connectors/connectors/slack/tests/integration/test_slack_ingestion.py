"""End-to-end: the Slack connector through the real cognee pipeline, Slack faked in memory.

Copied from cognee.tests.integration.tasks.connectors.test_slack_ingestion (cognee dev, 2026-11-12).

Covers what the unit tests cannot: the node set landing on the Data records once
namespaced, the channel container and ``child_of`` edges after cognify, an unchanged
sync that builds nothing, and a new reply that replaces only its own conversation.
"""

import cognee
import pytest
import pytest_asyncio
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from fake_slack import CHANNEL, TEAM, FakeSlack

from cognee_connector_slack import slack_source

DATASET = "slack_ingestion_test"
ROOT_A = "1793872860.000100"
REPLY_A = "1793872920.000100"
ROOT_B = "1793872980.000100"
# ROOT_B has no replies, so its document is its day's (G7).
DAY_B = f"{CHANNEL}:day:2026-11-05"


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.fixture
def mock_ai(monkeypatch):
    """Cognify over the journeys tier's deterministic LLM and embeddings."""
    from cognee.tests.journeys import mock_ai as mocks

    for name, value in {
        "LLM_API_KEY": "mock-key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "openai/gpt-5-mini",
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "openai/text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "256",
        "EMBEDDING_API_KEY": "mock-key",
        "COGNEE_SKIP_PREFLIGHT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config

    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()
    llm = mocks.install_all({})
    yield llm
    mocks.uninstall_all()
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()


def _channel() -> FakeSlack:
    fake = FakeSlack()
    fake.add_user("U1", "ana")
    fake.add_user("U2", "ben")
    fake.post(ROOT_A, "U1", "Ana asks where the Postgres migration plan is.")
    fake.reply(ROOT_A, REPLY_A, "U2", "Ben put it in the infra folder.")
    fake.post(ROOT_B, "U2", "Ben ships the release on Friday.")
    return fake


async def _sync(fake: FakeSlack) -> None:
    await cognee.add(
        slack_source(
            service=fake, team_id=TEAM, channel_id=CHANNEL, resource_name="slack_channel_test"
        ),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _records() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return {
        record.system_metadata["external_id"]: record
        for record in await get_dataset_data(datasets[0].id)
        if isinstance(record.system_metadata, dict)
        and record.system_metadata.get("source") == "slack"
    }


def _rate_limited_once(fake: FakeSlack, retry_after: int):
    """Slack over HTTP, served from ``fake``: the first ``conversations.history`` request
    answers 429 with ``Retry-After``, as Tier 1 history does under load."""
    import json
    from urllib.parse import parse_qsl

    import httpx

    limited = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        if method == "conversations.history" and not limited:
            limited.append(method)
            return httpx.Response(429, headers={"Retry-After": str(retry_after)})
        params = dict(parse_qsl(request.content.decode()))
        params.pop("token", None)
        return httpx.Response(200, content=json.dumps(fake.call(method, **params)))

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_slack_rate_limit_holds_up_no_other_datasets_staging(clean_environment):
    """Slack's rate limits are waited out inside ``pipeline.run``. Staging was one lock for
    the whole process, so a user's CSV upload or another connector waited behind Slack
    (review of #6683). While Slack waits, another dataset's staging is free."""
    # cognee's own staging lock names: this checks the host's ingestion, not the package.
    from cognee.tasks.ingestion import ingest_dlt_source
    from cognee.tasks.ingestion.dlt_utils import UNSCOPED_PIPELINE_NAME

    from cognee_connector_slack.source import SlackClient

    def free_for_another_dataset() -> bool:
        return not any(
            ingest_dlt_source._staging_lock(key).locked()
            for key in (f"pipeline:{UNSCOPED_PIPELINE_NAME}", "dataset:main_dataset")
        )

    while_waiting = []
    client = SlackClient(
        "xoxb-test",
        transport=_rate_limited_once(_channel(), retry_after=30),
        sleep=lambda seconds: while_waiting.append(
            (
                ingest_dlt_source._staging_lock(f"dataset:{DATASET}").locked(),
                free_for_another_dataset(),
            )
        ),
    )

    await cognee.add(
        slack_source(
            service=client, team_id=TEAM, channel_id=CHANNEL, resource_name="slack_channel_test"
        ),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # The wait happened with this dataset's staging held, and nobody else's.
    assert while_waiting == [(True, True)]


@pytest.mark.asyncio
async def test_a_sync_tags_each_conversation_with_its_channel(clean_environment):
    await _sync(_channel())

    records = await _records()
    assert set(records) == {f"{CHANNEL}:{ROOT_A}", DAY_B}
    for record in records.values():
        assert (record.external_metadata or {}).get("node_set") == [f"slack:channel:{CHANNEL}"]
        assert record.system_metadata["url"].startswith(
            f"https://acme.slack.com/archives/{CHANNEL}/p"
        )


@pytest.mark.asyncio
async def test_an_unchanged_sync_keeps_every_document(clean_environment):
    fake = _channel()
    await _sync(fake)
    before = {key: record.id for key, record in (await _records()).items()}

    await _sync(fake)

    assert {key: record.id for key, record in (await _records()).items()} == before


@pytest.mark.asyncio
async def test_a_new_reply_replaces_only_its_conversation(clean_environment):
    """The first reply turns ROOT_B into a thread: it leaves its day's document, which had
    nothing else and goes, and the other thread is not touched."""
    fake = _channel()
    await _sync(fake)
    before = {key: record.id for key, record in (await _records()).items()}

    fake.reply(ROOT_B, "1793873040.000100", "U1", "Ana confirms Friday.")
    await _sync(fake)

    after = {key: record.id for key, record in (await _records()).items()}
    assert set(after) == {f"{CHANNEL}:{ROOT_A}", f"{CHANNEL}:{ROOT_B}"}
    assert after[f"{CHANNEL}:{ROOT_A}"] == before[f"{CHANNEL}:{ROOT_A}"]


@pytest.mark.asyncio
async def test_cognify_places_every_conversation_under_its_channel(clean_environment, mock_ai):
    await _sync(_channel())
    await cognee.cognify(datasets=[DATASET])

    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    documents = {str(record.id): key for key, record in (await _records()).items()}
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
    properties = dict(nodes)

    def label(node_id) -> str:
        node_id = str(node_id)
        if node_id in documents:
            return documents[node_id]
        found = properties[node_id]
        assert found["type"] == "StructureContainer", found
        return f"{found['kind']}:{found['external_id']}:{found['name']}"

    container = f"slack_channel:{CHANNEL}:#general"
    assert {(label(e[0]), label(e[1])) for e in edges if e[2] == "child_of"} == {
        (f"{CHANNEL}:{ROOT_A}", container),
        (DAY_B, container),
    }


@pytest.mark.asyncio
async def test_an_at_cognee_answer_names_the_slack_conversation_it_read(clean_environment, mock_ai):
    """G10: the search ``@cognee`` runs names each document it read by the row's title and
    links it to the message, and the answer shows that as its source."""
    from cognee.modules.search.types import SearchType

    from cognee_connector_slack.cognee_host.handle_cognee_ask import _format_answer

    await _sync(_channel())
    await cognee.cognify(datasets=[DATASET])

    results = await cognee.search(
        "Where is the Postgres migration plan?",
        query_type=SearchType.HYBRID_COMPLETION,
        user=await get_default_user(),
        datasets=None,
        include_references=True,
        verbose=True,
    )

    _, blocks = _format_answer(results)
    sources = [block for block in blocks if block["type"] == "context"]
    permalink = f"https://acme.slack.com/archives/{CHANNEL}/p{ROOT_A.replace('.', '')}"
    assert sources, blocks
    # SDK-1502: where, who and when, read from what the row already stores: the channel's
    # name from its structure (never hashed), the author and day from its title.
    from datetime import datetime, timezone

    day = "Nov 5" if datetime.now(timezone.utc).year == 2026 else "Nov 5, 2026"
    assert f"<{permalink}|#general · ana · {day}>" in sources[0]["elements"][0]["text"]


# -- Review 2026-11-11 (F4): red until fixed ---------------------------------------------

PRIVATE = "G0LEADS"
ASKER = "U0ASKER"


@pytest.mark.asyncio
async def test_a_member_is_not_answered_from_a_private_channel_they_are_not_in(
    clean_environment, mock_ai, monkeypatch
):
    """F4: every selected channel lands in the installer's one workspace dataset, and
    ``@cognee`` searched every dataset the asker can read. Once the installer shared that
    dataset, a member got answers from every selected private channel, a member of it or
    not. Slack's own rule must hold: a member's direct message is answered only from
    channels they can see."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cognee.modules.users.methods import create_user
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    from cognee_connector_slack.cognee_host import handle_cognee_mention as mention

    fake = _channel()
    fake.add_channel(PRIVATE, "leads", is_private=True)
    fake.members[PRIVATE] = ["U1"]  # the asker is not in it
    fake.members[CHANNEL] = ["U1", ASKER]
    fake.post("1793873100.000100", "U1", "Leads only: the salary review moves to May.", PRIVATE)
    await _sync(fake)
    await cognee.add(
        slack_source(
            service=fake, team_id=TEAM, channel_id=PRIVATE, resource_name="slack_channel_leads"
        ),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])

    installer = await get_default_user()
    member = await create_user("asker@example.com", "pw-Test-123")
    (dataset,) = await get_authorized_existing_datasets(
        user=installer, permission_type="read", datasets=[DATASET]
    )
    await give_permission_on_dataset(member, dataset.id, "read")

    credential = SimpleNamespace(
        status="active",
        provider_account_id=TEAM,
        user_id=installer.id,
        provider_metadata={"bot_user_id": "U0BOT", "selected_channels": [CHANNEL, PRIVATE]},
        last_synced_at=None,
        sync_status=None,
    )
    replies, answers = [], []

    async def post_ephemeral(credential, channel, user, payload, thread_ts=None):
        replies.append(payload)

    async def post_message(credential, channel, payload, thread_ts=None):
        answers.append(payload)

    monkeypatch.setattr(mention, "get_by_team", AsyncMock(return_value=credential))
    monkeypatch.setattr(mention, "resolve_owner_user_id", AsyncMock(return_value=member.id))
    monkeypatch.setattr(mention, "post_ephemeral", post_ephemeral)
    monkeypatch.setattr(mention, "post_message", post_message)
    # Slack's membership answer (users.conversations) and the connection's dataset name
    # (this module syncs into its own dataset) are the seams.
    monkeypatch.setattr(
        mention,
        "member_channels",
        AsyncMock(return_value={c for c, members in fake.members.items() if ASKER in members}),
    )
    monkeypatch.setattr(mention, "dataset_name_for_credential", lambda credential: DATASET)
    event = {
        "type": "message",
        "channel_type": "im",
        "user": ASKER,
        "channel": "D0ASKER",
        "ts": "1793873200.000100",
        "text": "when is the salary review?",
    }
    await mention.handle_cognee_mention(TEAM, event)

    assert answers, "the direct message is answered from the channel the asker is in"
    shown = " ".join(str(payload) for payload in [*replies, *answers])
    assert PRIVATE not in shown
    assert "salary review moves to May" not in shown


# -- Public answers in a channel (SDK-1502) ----------------------------------------------

PUBLIC = "C0DESIGN"


@pytest.mark.asyncio
async def test_a_channel_answer_uses_only_what_everyone_in_the_channel_can_see(
    clean_environment, mock_ai, monkeypatch
):
    """Goran, 2026-11-12: a question in a channel is answered in its thread, for everyone
    there, so only that channel and the selected public channels may answer it. The asker
    is in the private channel and owns a dataset of their own, and today's private answer
    reads both; the public one must not, or the channel reads what only the asker may."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from cognee.modules.users.methods import create_user
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    from cognee_connector_slack.cognee_host import channel_membership as membership
    from cognee_connector_slack.cognee_host import handle_cognee_mention as mention

    fake = _channel()
    fake.add_channel(PUBLIC, "design")
    fake.add_channel(PRIVATE, "leads", is_private=True)
    fake.members[PRIVATE] = ["U1", ASKER]  # the asker is in it; the channel asked in is wider
    fake.members[CHANNEL] = ["U1", "U2", ASKER]
    fake.post("1793873000.000100", "U2", "Design: the review of the logo is on Monday.", PUBLIC)
    fake.post(
        "1793873100.000100", "U1", "Leads only: the review of salaries moves to May.", PRIVATE
    )
    await _sync(fake)
    for channel, name in ((PUBLIC, "slack_channel_design"), (PRIVATE, "slack_channel_leads")):
        await cognee.add(
            slack_source(service=fake, team_id=TEAM, channel_id=channel, resource_name=name),
            dataset_name=DATASET,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    await cognee.cognify(datasets=[DATASET])

    installer = await get_default_user()
    member = await create_user("asker@example.com", "pw-Test-123")
    (dataset,) = await get_authorized_existing_datasets(
        user=installer, permission_type="read", datasets=[DATASET]
    )
    await give_permission_on_dataset(member, dataset.id, "read")
    await cognee.add(
        "Personal: the review of my own bonus is on Sunday.",
        dataset_name="asker_notes",
        user=member,
    )
    await cognee.cognify(datasets=["asker_notes"], user=member)

    credential = SimpleNamespace(
        status="active",
        provider_account_id=TEAM,
        user_id=installer.id,
        provider_metadata={
            "bot_user_id": "U0BOT",
            "selected_channels": [CHANNEL, PUBLIC, PRIVATE],
        },
        last_synced_at=None,
        sync_status=None,
    )
    private, public = [], []

    async def post_ephemeral(credential, channel, user, payload, thread_ts=None):
        private.append(payload)

    async def post_message(credential, channel, payload, thread_ts=None):
        public.append((channel, thread_ts, payload))

    monkeypatch.setattr(mention, "get_by_team", AsyncMock(return_value=credential))
    monkeypatch.setattr(mention, "resolve_owner_user_id", AsyncMock(return_value=member.id))
    monkeypatch.setattr(mention, "post_ephemeral", post_ephemeral)
    monkeypatch.setattr(mention, "post_message", post_message)
    # Slack's answers (which channels the asker is in, which are public) and the
    # connection's dataset name (this module syncs into its own dataset) are the seams.
    monkeypatch.setattr(
        mention,
        "member_channels",
        AsyncMock(return_value={c for c, members in fake.members.items() if ASKER in members}),
    )
    monkeypatch.setattr(
        membership,
        "public_channels",
        AsyncMock(return_value={c for c, info in fake.channels.items() if not info["is_private"]}),
    )
    monkeypatch.setattr(
        membership, "shared_with_other_organizations", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(membership, "is_guest", AsyncMock(return_value=False))
    monkeypatch.setattr(mention, "dataset_name_for_credential", lambda credential: DATASET)
    event = {
        "type": "app_mention",
        "user": ASKER,
        "channel": CHANNEL,
        "ts": "1793873200.000100",
        "text": "<@U0BOT> when is the review?",
    }
    await mention.handle_cognee_mention(TEAM, event)

    shown = " ".join(str(payload) for payload in [*private, *public])
    assert "salaries moves to May" not in shown and PRIVATE not in shown
    assert "bonus is on Sunday" not in shown
    assert len(public) == 1, "one answer, posted to the channel"
    ((channel, thread_ts, answer),) = public
    assert (channel, thread_ts) == (CHANNEL, event["ts"])
    assert "logo is on Monday" in str(answer), "the selected public channel answers"
