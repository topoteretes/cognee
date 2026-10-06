"""``codegraph_config={"treat_as_repository": True}`` clones what detection won't claim.

``add()``'s own repository detection is deliberately narrow: ``code_repo_clone_url``
claims GitHub/GitLab roots and ``.git`` URLs, and ``detect_code_project`` claims
directories with a build manifest or ``.git``. It has to be, because most http(s)
URLs really are web pages — a wider guess would scrape a repository or, worse,
clone a web page.

The removed ``content_type="code"`` was the explicit declaration that made the
guessing unnecessary: it took every item as a repository spec and handed it to
``resolve_repo_source``, which clones any git remote. ``treat_as_repository``
is that declaration as a first-class option, so the deprecated value still
means what it meant and new callers have a supported way to say it.

Regression guard: without it, ``remember("https://bitbucket.org/o/r",
content_type="code")`` silently fetched and stored the project's web page
instead of building its code graph.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

code_repo = importlib.import_module("cognee.tasks.code_graph.code_repo")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")

# Real repository URLs that add()'s detection does not claim: cloning them is
# only correct because the caller said they are repositories.
UNDETECTED_REMOTES = [
    "https://bitbucket.org/acme/api",
    "https://gitlab.acme.com/team/api",
    "https://git.acme.io/team/api",
    "git@github.com:acme/api.git",
    "ssh://git@git.acme.com/team/api",
]


@pytest.fixture
def cloned(monkeypatch, tmp_path):
    """Record what resolve_repo_source is asked for; hand back a fake clone."""
    seen = []

    async def fake_resolve_repo_source(spec, clones_dir=None, credentials=None):
        seen.append((str(spec), credentials))
        clone = tmp_path / "clone"
        clone.mkdir(exist_ok=True)
        return clone

    async def fake_resolve_code_repository(directory, **kwargs):
        item = SimpleNamespace(
            data=f"manifest:{directory}",
            system_metadata={
                "source": "code_repo",
                **({"repo_url": kwargs["source_url"]} if kwargs.get("source_url") else {}),
            },
            kwargs=kwargs,
        )
        return item, [Path(directory) / "README.md"], 0

    monkeypatch.setattr(code_repo, "resolve_repo_source", fake_resolve_repo_source)
    monkeypatch.setattr(code_repo, "resolve_code_repository", fake_resolve_code_repository)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", UNDETECTED_REMOTES)
async def test_a_declared_remote_is_cloned_not_sniffed(cloned, spec):
    resolved = await resolve_data_directories([spec], treat_as_repository=True)

    assert cloned == [(spec, None)], "the spec never reached the cloner"
    assert len(resolved) == 2
    assert resolved[0].system_metadata["source"] == "code_repo"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", UNDETECTED_REMOTES)
async def test_without_the_declaration_the_same_spec_is_left_alone(cloned, spec):
    # The contrast that makes the option necessary. An ssh spec is refused
    # outright (it can only be a repository); the http(s) ones fall through to
    # the web-page path, which is right for a URL nobody declared.
    if spec.startswith(("git@", "ssh://")):
        with pytest.raises(IngestionError, match="ssh git remotes"):
            await resolve_data_directories([spec])
        return
    assert await resolve_data_directories([spec]) == [spec]
    assert cloned == []


@pytest.mark.asyncio
async def test_a_declared_directory_needs_no_project_marker(cloned, tmp_path):
    # detect_code_project() would reject this tree (no .git, no build manifest).
    plain = tmp_path / "sources"
    (plain / "pkg").mkdir(parents=True)
    (plain / "pkg" / "a.py").write_text("def a():\n    return 1\n")

    resolved = await resolve_data_directories([str(plain)], treat_as_repository=True)

    assert cloned == [(str(plain), None)]
    assert resolved[0].system_metadata["source"] == "code_repo"
    # A local repository is not cognee's own clone, so its documents stay
    # caller-supplied path strings and keep the ACCEPT_LOCAL_FILE_PATH gate.
    assert all(isinstance(item, str) for item in resolved[1:])


@pytest.mark.asyncio
async def test_a_declared_clone_marks_its_documents_as_cognees_own(cloned):
    from cognee.tasks.ingestion.repo_clone_file import RepoCloneFile

    resolved = await resolve_data_directories(
        ["https://bitbucket.org/acme/api"], treat_as_repository=True
    )

    assert all(isinstance(item, RepoCloneFile) for item in resolved[1:])


@pytest.mark.asyncio
async def test_credentials_reach_the_cloner(cloned):
    await resolve_data_directories(
        ["https://git.acme.com/team/api"], credentials="tok123", treat_as_repository=True
    )

    assert cloned == [("https://git.acme.com/team/api", "tok123")]


@pytest.mark.asyncio
async def test_a_url_credential_is_redacted_before_it_is_stored(cloned):
    spec = "https://x-access-token:secret@git.acme.com/team/api"

    resolved = await resolve_data_directories([spec], treat_as_repository=True)

    stored = resolved[0].system_metadata["repo_url"]
    assert "secret" not in stored
    assert stored == "https://git.acme.com/team/api"


@pytest.mark.asyncio
async def test_include_documents_false_still_suppresses_them(cloned):
    resolved = await resolve_data_directories(
        ["https://bitbucket.org/acme/api"], include_documents=False, treat_as_repository=True
    )

    assert resolved[0].kwargs["include_documents"] is False


@pytest.mark.asyncio
async def test_a_non_spec_item_raises_instead_of_being_stored_as_text(cloned):
    # The declaration says every item is a repository; an upload among them is
    # a mistake, and storing it silently is what the old route refused to do.
    with pytest.raises(IngestionError, match="treat_as_repository expects repository paths"):
        await resolve_data_directories(
            [SimpleNamespace(file=b"", filename="notes.txt")], treat_as_repository=True
        )


@pytest.mark.asyncio
async def test_local_specs_are_still_gated_by_accept_local_file_path(cloned, monkeypatch, tmp_path):
    # Declaring something a repository is not a licence to read a disabled root.
    from cognee.tasks.ingestion.save_data_item_to_storage import settings as save_data_settings

    monkeypatch.setattr(save_data_settings, "accept_local_file_path", False)

    with pytest.raises(IngestionError, match="ACCEPT_LOCAL_FILE_PATH=false"):
        await resolve_data_directories([str(tmp_path)], treat_as_repository=True)

    # A remote spec is unaffected: nothing is read from this machine.
    assert await resolve_data_directories(
        ["https://bitbucket.org/acme/api"], treat_as_repository=True
    )


@pytest.mark.asyncio
async def test_the_deprecated_content_type_still_declares_its_repositories(monkeypatch):
    """The end-to-end reason this option exists."""
    calls = {}

    async def fake_add(*args, **kwargs):
        calls["add"] = kwargs

    async def fake_cognify(*args, **kwargs):
        calls["cognify"] = kwargs
        return {}

    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)
    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)

    await remember_module.remember(
        "https://bitbucket.org/acme/api",
        dataset_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        self_improvement=False,
        content_type="code",
    )

    assert calls["add"]["codegraph_config"]["treat_as_repository"] is True
