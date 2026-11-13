"""This package's source against cognee's in-tree copy, run for run, on the same fake Dropbox.

Dropbox lives twice until the in-tree copy is retired (SDK-1262): as
``cognee.tasks.ingestion.connectors.dropbox`` in cognee and as this package. Both must yield
the same rows (ids, titles, text, structure, tombstones), the same ``cognee_sync_stats`` and the
same dlt state, or a host that switches from one to the other re-extracts, duplicates or
forgets documents. Each step changes the fake once and runs both copies on it, each with its
own state carried over from its previous run.

Skipped when the installed cognee has no in-tree Dropbox connector (cognee releases before it
shipped). The connectors CI installs cognee from this checkout, so the comparison runs there.
"""

import httpx
import pytest
from fake_dropbox import ACCOUNT, TOKEN, FakeDropbox, lookup_error, make_pdf

from cognee_connector_dropbox import source as package

in_tree = pytest.importorskip(
    "cognee.tasks.ingestion.connectors.dropbox",
    reason="the installed cognee has no in-tree Dropbox connector to compare with",
)


class MixedCaseDropbox(FakeDropbox):
    """Ids in the mixed case Dropbox issues (``id:a4ayc_80_OEAAAAAAAAAXw``), so a copy that
    changed an id's case would yield another document id than the other copy."""

    def _new_id(self, kind):
        return super()._new_id(kind) + "_OEAAAAAAAAAXw"


def _run(module, fake, state, folder_id, max_mb):
    client = module.DropboxClient(
        TOKEN,
        http=httpx.Client(transport=httpx.MockTransport(fake.handle)),
        sleep=lambda _seconds: None,
    )
    stats = dict.fromkeys(module.STATS_KEYS, 0)
    try:
        rows = list(
            module._iter_rows(
                module._Session(client, None),
                folder_id=folder_id,
                account_id=ACCOUNT,
                state=state,
                stats=stats,
                max_bytes=None if max_mb is None else max_mb * 1024 * 1024,
            )
        )
        error = None
    except module.DropboxError as raised:
        rows, error = [], (type(raised).__name__, raised.status, raised.endpoint, raised.tags)
    finally:
        client.close()
    return rows, stats, error


class Twins:
    """The two copies, each with its own state, synced one after the other on one fake."""

    def __init__(self, fake, folder_id="", max_mb=None):
        self.fake = fake
        self.folder_id = folder_id
        self.max_mb = max_mb
        self.states = {"in_tree": {}, "package": {}}
        self.runs = 0

    def sync(self, prepare=None):
        """Run both copies once and assert they agree; return the package's rows and stats.

        ``prepare`` runs before each copy, for a one-shot answer the fake consumes.
        """
        results = {}
        for name, module in (("in_tree", in_tree), ("package", package)):
            if prepare is not None:
                prepare()
            calls = len(self.fake.calls)
            rows, stats, error = _run(
                module, self.fake, self.states[name], self.folder_id, self.max_mb
            )
            results[name] = (rows, stats, error, self.fake.calls[calls:])
        self.runs += 1
        (rows, stats, error, calls), (p_rows, p_stats, p_error, p_calls) = (
            results["in_tree"],
            results["package"],
        )
        assert p_rows == rows, f"run {self.runs}: rows differ"
        assert p_stats == stats, f"run {self.runs}: stats differ"
        assert p_error == error, f"run {self.runs}: errors differ"
        assert p_calls == calls, f"run {self.runs}: Dropbox requests differ"
        assert self.states["package"] == self.states["in_tree"], f"run {self.runs}: state differs"
        return p_rows, p_stats


def test_both_copies_share_their_public_constants():
    for name in (
        "STATS_KEYS",
        "TEXT_EXTENSIONS",
        "PDF_EXTENSION",
        "TEXT_EXPORT_FORMATS",
        "LIST_FOLDER_LIMIT",
        "RUN_WAIT_BUDGET_SECONDS",
        "RUN_TIME_BUDGET_SECONDS",
        "RETRIES",
        "WAIT_STEP_SECONDS",
        "DOWNLOAD_SPOOL_BYTES",
        "_STATE_VERSION",
        "_RENDER_VERSION",
        "_API",
        "_CONTENT",
    ):
        assert getattr(package, name) == getattr(in_tree, name), name


def test_both_factories_declare_the_same_resource():
    pytest.importorskip("dlt")
    arguments = {
        "folder_id": "id:abc",
        "account_id": ACCOUNT,
        "resource_name": "dropbox_notes",
        "token": TOKEN,
        "max_file_size_mb": 5,
    }
    ours, theirs = package.dropbox_source(**arguments), in_tree.dropbox_source(**arguments)

    assert ours.name == theirs.name
    assert ours.write_disposition == theirs.write_disposition
    assert ours.columns == theirs.columns
    assert ours._hints.get("primary_key") == theirs._hints.get("primary_key")
    for attribute in ("cognee_document_source", "cognee_pipeline_scope", "cognee_sync_stats"):
        assert getattr(ours, attribute) == getattr(theirs, attribute), attribute


def test_a_selected_folder_gives_the_same_rows_stats_and_state_through_its_whole_life():
    fake = MixedCaseDropbox(page_size=3)
    notes = fake.mkdir("/Notes")
    fake.mkdir("/Notes/2026")
    plan = fake.put("/Notes/2026/plan.md", "# Plan\n\nShip the connector.")
    fake.put("/Notes/todo.txt", "buy milk")
    fake.put("/Notes/table.csv", "a,b\n1,2")
    report = fake.put("/Notes/report.pdf", make_pdf("Quarterly report Zephyrcorp"))
    fake.put("/Notes/photo.png", b"\x89PNG")
    fake.put("/Notes/empty.md", "")
    fake.put("/Notes/binary.csv", b"a,b\x00\x01")
    fake.put("/Notes/notepad.txt", "hello from notepad".encode("utf-16"))
    fake.put("/Notes/huge.txt", "x", size=2 * 1024 * 1024)
    fake.put(
        "/Notes/Roadmap.paper",
        "# Roadmap",
        downloadable=False,
        export_info={"export_as": "markdown", "export_options": ["markdown", "html"]},
    )
    fake.put("/Notes/Sheet.gsheet", "a", downloadable=False, export_info={"export_as": "xlsx"})
    fake.put("/Notes/undated.md", "no date", modified=None)
    fake.put("/Elsewhere/c.md", "not selected")
    twins = Twins(fake, notes, max_mb=1)

    rows, stats = twins.sync()
    assert plan in {row["id"] for row in rows} and report in {row["id"] for row in rows}
    assert stats["complete"] == 1 and stats["skipped_too_large"] == 1

    twins.sync()  # unchanged

    fake.put("/Notes/2026/plan.md", "# Plan\n\nShipped.", modified="2026-03-02T10:00:00Z")
    fake.put("/Notes/report.pdf", make_pdf("Quarterly report Zephyrcorp") + b"\n% annotated")
    fake.move("/Notes/todo.txt", "/Notes/2026/todo.txt")
    fake.move("/Notes/table.csv", "/Elsewhere/table.csv")
    fake.put("/Notes/empty.md", "now it has text")
    twins.sync()

    fake.move("/Notes/2026", "/Notes/2027")
    fake.delete("/Notes/notepad.txt")
    twins.sync()

    fake.download_errors[plan] = (500, {})
    fake.put("/Notes/2027/plan.md", "# Plan\n\nRetry me.")
    rows, stats = twins.sync()
    assert stats["failed_retry"] == 1
    del fake.download_errors[plan]
    twins.sync()

    fake.download_errors[report] = lookup_error("restricted_content")
    fake.put("/Notes/report.pdf", make_pdf("Restricted now"))
    twins.sync()

    fake.reset_cursors.add(twins.states["package"]["cursor"])
    fake.put("/Notes/after-reset.md", "listed whole again")
    twins.sync()

    fake.delete("/Notes/2027")
    twins.sync()

    fake.delete("/Notes")
    rows, stats = twins.sync()
    assert rows and all(row["_deleted"] for row in rows)
    rows, stats = twins.sync()
    assert rows == [] and stats["failed_folder_gone"] == 1


def test_the_whole_dropbox_gives_the_same_rows_and_stats():
    fake = MixedCaseDropbox(page_size=2)
    for i in range(7):
        fake.put(f"/Dir{i % 3}/f{i}.md", f"file {i}")
    twins = Twins(fake, "")
    twins.sync()

    fake.vanish("/Dir0/f0.md")
    fake.put("/Dir1/f1.md", "file 1, edited")
    fake.delete("/Dir2")
    twins.sync()


def test_failures_are_the_same_failures():
    fake = MixedCaseDropbox()
    fake.put("/Notes/a.md", "alpha")

    # A folder id never listed that Dropbox does not find.
    Twins(fake, "id:missing").sync()

    # A token of another account.
    fake.account = "dbid:someone-else"
    Twins(fake, "").sync()
    fake.account = ACCOUNT

    # A rate limit while listing cuts the run; the next run lists again.
    twins = Twins(fake, "")
    _rows, stats = twins.sync(
        prepare=lambda: fake.queued.update(
            {"files/list_folder": [(429, {}, {"Retry-After": "3600"})]}
        )
    )
    assert stats["failed_rate_limit"] == 1
    twins.sync()


def test_a_team_space_is_reported_the_same_way():
    fake = MixedCaseDropbox()
    fake.team_space = True
    fake.put("/a.md", "alpha")
    _rows, stats = Twins(fake, "").sync()
    assert stats["team_space_unsupported"] == 1


SELECTIONS = {
    "none": ([], []),
    "one": (["id:a"], [{"path_lower": "/a"}]),
    "one gone": (["id:gone"], [None]),
    "siblings": (["id:a", "id:b"], [{"path_lower": "/notes"}, {"path_lower": "/notes-old"}]),
    "nested": (["id:a", "id:b"], [{"path_lower": "/notes"}, {"path_lower": "/notes/sub"}]),
    "one of two gone": (["id:a", "id:gone"], [{"path_lower": "/notes"}, None]),
}


async def _nesting_check(refuse, client_module, args, folder_ids, metas, refuse_missing):
    """The Dropbox lookups one copy's refuse_nested_selection makes and how it ends."""
    from unittest.mock import AsyncMock, patch

    lookup = AsyncMock(return_value=metas)
    with patch.object(client_module, "get_folders", lookup):
        try:
            await refuse(*args, folder_ids, refuse_missing=refuse_missing)
            outcome = None
        except ValueError as error:
            outcome = type(error).__name__
    return [call.args[1] for call in lookup.await_args_list], outcome


@pytest.mark.parametrize("refuse_missing", [False, True])
@pytest.mark.parametrize("case", list(SELECTIONS))
@pytest.mark.asyncio
async def test_the_nesting_check_asks_and_refuses_the_same(case, refuse_missing):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from cognee_connector_dropbox import client as package_client
    from cognee_connector_dropbox import selection

    in_tree_sync = pytest.importorskip("cognee.modules.integrations.dropbox.sync")
    folder_ids, metas = SELECTIONS[case]
    owner = SimpleNamespace(provider="dropbox", provider_account_id=ACCOUNT)

    with patch.object(
        in_tree_sync.ingestion, "require_active_credential", AsyncMock(return_value=owner)
    ):
        cognees = await _nesting_check(
            in_tree_sync.refuse_nested_selection,
            in_tree_sync.client,
            (owner, TOKEN),
            folder_ids,
            metas,
            refuse_missing,
        )
    packages = await _nesting_check(
        selection.refuse_nested_selection,
        package_client,
        (TOKEN,),
        folder_ids,
        metas,
        refuse_missing,
    )

    assert packages == cognees, case
