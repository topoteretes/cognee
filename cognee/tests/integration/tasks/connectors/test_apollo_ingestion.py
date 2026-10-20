# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A stored Apollo document is replaced only when its rendered text changes.

The source runs through the real cognee.add() pipeline with the API faked. ``remember()`` runs
this same add() step first and then cognify(), which needs an LLM, so this stops at the Data
records like the Linear test. A deleted record leaves the dataset, and neither the API key nor
a custom field nobody listed lands in a stored document.
"""

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import apollo_source
from cognee.tasks.ingestion.connectors.apollo import (
    BASE_URL,
    ApolloNotFoundError,
    ApolloRateLimitedError,
)
from cognee.tests.unit.tasks.connectors.fake_apollo import FakeApollo

DATASET = "apollo_integration_test"
TABLE = "apollo_test"
KEY = "SECRET-apollo-key-123"
PHONE = "+1 555 0199"
# Calls a first sync makes before it reaches the contacts search: contact stages, account
# stages, lists, the sequence list and the accounts page.
CALLS_BEFORE_CONTACTS = 5


def _workspace() -> FakeApollo:
    fake = FakeApollo()
    fake.add_sequence("s1", "Q4 Outreach")
    fake.add_account("a1", "Acme")
    fake.add_contact("c1", "Ada Lovelace", account_id="a1")
    fake.add_contact("c2", "Ben Turing", account_id="a1")
    return fake


async def _add(source) -> None:
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _sync(fake: FakeApollo, resource_name: str = TABLE, **kwargs):
    source = apollo_source(service=fake, resource_name=resource_name, **kwargs)
    await _add(source)
    return source


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        data.system_metadata["external_id"]: data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "apollo"
    }


def _stored_text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_tagged_documents_and_a_quiet_one_changes_nothing(
    clean_environment,
):
    fake = _workspace()
    await _sync(fake)
    first = await _documents()

    assert set(first) == {"sequence:s1", "account:a1", "contact:c1", "contact:c2"}
    assert {doc.system_metadata["table_name"] for doc in first.values()} == {TABLE}

    quiet = await _sync(fake)
    assert quiet.cognee_sync_stats["skipped"] == 4
    assert {k: d.id for k, d in (await _documents()).items()} == {k: d.id for k, d in first.items()}


@pytest.mark.asyncio
async def test_an_enrollment_replaces_only_that_contacts_document(clean_environment):
    fake = _workspace()
    await _sync(fake)
    before = {k: d.id for k, d in (await _documents()).items()}

    fake.enroll("c1", "s1")
    await _sync(fake)
    after = {k: d.id for k, d in (await _documents()).items()}

    assert set(after) == set(before)
    assert after["contact:c1"] != before["contact:c1"]
    assert {k: v for k, v in after.items() if k != "contact:c1"} == {
        k: v for k, v in before.items() if k != "contact:c1"
    }


@pytest.mark.asyncio
async def test_a_sequence_step_replaces_no_document(clean_environment):
    """A step a sequence sends moves the contact's activity, not what its document says, so
    it must not cost the contact a new extraction."""
    fake = _workspace()
    fake.enroll("c1", "s1")
    await _sync(fake)
    before = {k: d.id for k, d in (await _documents()).items()}

    fake.advance_step("c1")
    again = await _sync(fake)

    assert again.cognee_sync_stats["skipped"] == again.cognee_sync_stats["scanned"] == 4
    assert {k: d.id for k, d in (await _documents()).items()} == before


@pytest.mark.asyncio
async def test_a_contact_deleted_in_apollo_is_forgotten(clean_environment):
    fake = _workspace()
    await _sync(fake)

    del fake.contacts["c2"]
    await _sync(fake)

    assert set(await _documents()) == {"sequence:s1", "account:a1", "contact:c1"}


@pytest.mark.asyncio
async def test_a_rate_limited_first_sync_resumes_without_losing_documents(clean_environment):
    fake = _workspace()
    fake.fail_after = CALLS_BEFORE_CONTACTS
    fake.fail_with = ApolloRateLimitedError("Apollo rate limit exceeded (HTTP 429)")
    cut = await _sync(fake)
    assert cut.cognee_sync_stats["failed_rate_limit"] == 1
    assert set(await _documents()) == {"sequence:s1", "account:a1"}

    fake.fail_after = None
    done = await _sync(fake)
    assert done.cognee_sync_stats["failed"] == 0
    assert set(await _documents()) == {"sequence:s1", "account:a1", "contact:c1", "contact:c2"}


class _HTTPResponse:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload
        self.headers = {"x-minute-requests-left": "49", "x-hourly-requests-left": "199"}

    def json(self):
        return self._payload


def _serve(fake: FakeApollo, keys: list[str]):
    """Route requests.request into the fake, the way the real HTTP API would answer."""

    def request(method, url, *, json=None, params=None, headers=None, timeout=None):
        keys.append(headers["x-api-key"])
        try:
            payload = fake.request(method, url.removeprefix(BASE_URL), body=json, params=params)
        except ApolloNotFoundError:
            return _HTTPResponse(422, {"error": "This contact has been deleted."})
        return _HTTPResponse(200, payload)

    return request


@pytest.mark.asyncio
async def test_the_real_client_syncs_and_the_api_key_never_reaches_stored_documents(
    clean_environment, monkeypatch
):
    fake, keys = _workspace(), []
    monkeypatch.setattr("requests.request", _serve(fake, keys))
    source = apollo_source(api_key=KEY, resource_name=TABLE)
    assert KEY not in repr(source)

    await _add(source)
    del fake.contacts["c2"]
    await _add(apollo_source(api_key=KEY, resource_name=TABLE))

    documents = await _documents()
    assert set(documents) == {"sequence:s1", "account:a1", "contact:c1"}
    assert keys and set(keys) == {KEY}
    for doc in documents.values():
        assert KEY not in repr(doc)
        assert KEY not in _stored_text(doc)


@pytest.mark.asyncio
async def test_a_custom_field_value_is_stored_only_when_its_id_is_listed(clean_environment):
    fake = _workspace()
    fake.contacts["c1"]["typed_custom_fields"] = {"f1": PHONE}

    await _sync(fake)
    assert all(PHONE not in _stored_text(doc) for doc in (await _documents()).values())

    await _sync(fake, custom_field_ids=["contact.f1"])
    assert PHONE in _stored_text((await _documents())["contact:c1"])

    # dropping the field from the list replaces the document again, without the value
    await _sync(fake)
    assert PHONE not in _stored_text((await _documents())["contact:c1"])


@pytest.mark.asyncio
async def test_two_apollo_accounts_in_one_dataset_stay_apart(clean_environment):
    team_a, team_b = _workspace(), FakeApollo(team_id="team2")
    team_b.add_contact("b1", "Dev Hollerith")
    team_b.add_contact("b2", "Eve Shannon")

    await _sync(team_a, "apollo_a")
    await _sync(team_b, "apollo_b")
    both = {"sequence:s1", "account:a1", "contact:c1", "contact:c2", "contact:b1", "contact:b2"}
    assert set(await _documents()) == both

    del team_b.contacts["b2"]
    await _sync(team_b, "apollo_b")
    await _sync(team_a, "apollo_a")

    assert set(await _documents()) == both - {"contact:b2"}
