"""Unit tests for incremental re-cognify (#3669): incremental_update + `cognee hook`.

Deterministic and offline — the graph/LLM seams (add, cognify, dataset lookup,
delete path) are mocked, so no API keys or databases are needed. What's exercised is
the new logic: removed-source detection, its safety scoping, and the git hook writer.
"""

import argparse
import asyncio
import importlib
import os
from pathlib import Path

# Load the submodule directly: the `update` package rebinds the name
# `incremental_update` to the function, which would shadow the module on a plain
# `import ... as iu`.
iu = importlib.import_module("cognee.api.v1.update.incremental_update")
from cognee.cli.commands.hook_command import HookCommand, _MARK_START, _MARK_END


class _Rec:
    def __init__(self, rec_id, location):
        self.id = rec_id
        self.original_data_location = location


class _Dataset:
    id = "ds-1"


class _User:
    id = "user-1"


def _run(coro):
    return asyncio.run(coro)


def _patch(monkeypatch_targets):
    """Set attributes on the incremental_update module; return a restore callable."""
    saved = {name: getattr(iu, name) for name in monkeypatch_targets}
    for name, value in monkeypatch_targets.items():
        setattr(iu, name, value)

    def restore():
        for name, value in saved.items():
            setattr(iu, name, value)

    return restore


# --------------------------------------------------------------------------- #
# _normalize_path
# --------------------------------------------------------------------------- #
def test_normalize_path_passthrough_and_abspath():
    assert iu._normalize_path("s3://bucket/key") == "s3://bucket/key"  # not a local path
    assert iu._normalize_path("/a/b/c") == os.path.normcase(os.path.abspath("/a/b/c"))


# --------------------------------------------------------------------------- #
# incremental_update: pruning
# --------------------------------------------------------------------------- #
def _run_update(tmp_files, recorded, *, prune=True, dataset_exists=True):
    """Drive incremental_update against a real temp dir with mocked graph seams.

    ``tmp_files`` = existing filenames under a fresh root dir. Returns
    (result, deleted_ids, add_calls, cognify_calls).
    """
    import tempfile

    root = tempfile.mkdtemp(prefix="iu_test_")
    for name in tmp_files:
        Path(root, name).write_text("content", encoding="utf-8")

    deleted = []
    calls = {"add": 0, "cognify": 0}

    async def _setup():
        pass

    async def _add(**kwargs):
        calls["add"] += 1

    async def _cognify(**kwargs):
        calls["cognify"] += 1
        return {"ok": True}

    async def _get_default_user():
        return _User()

    async def _get_authorized(datasets, permission_type, user):
        return [_Dataset()] if dataset_exists else []

    async def _get_dataset_data(dataset_id):
        # Resolve relative record locations against the temp root.
        return [
            _Rec(r.id, r.original_data_location if os.path.isabs(str(r.original_data_location))
                 else os.path.join(root, str(r.original_data_location)))
            for r in recorded
        ]

    async def _has_nodes(dataset_id, data_id):
        return True

    async def _delete_nodes(dataset_id, data_id, user_id):
        pass

    async def _delete_data(record, dataset_id):
        deleted.append(record.id)

    restore = _patch(
        {
            "setup": _setup,
            "add": _add,
            "cognify": _cognify,
            "get_default_user": _get_default_user,
            "get_authorized_existing_datasets": _get_authorized,
            "get_dataset_data": _get_dataset_data,
            "has_data_related_nodes": _has_nodes,
            "delete_data_nodes_and_edges": _delete_nodes,
            "delete_data": _delete_data,
        }
    )
    try:
        result = _run(
            iu.incremental_update(root, dataset_name="ds", prune_removed=prune)
        )
    finally:
        restore()
    return result, deleted, calls


def test_prunes_only_removed_source_under_root():
    # present.txt exists on disk; gone.txt is recorded but absent -> only gone is pruned.
    result, deleted, calls = _run_update(
        tmp_files=["present.txt"],
        recorded=[_Rec("keep", "present.txt"), _Rec("drop", "gone.txt")],
    )
    assert deleted == ["drop"]
    assert result["removed"] == 1
    assert calls["add"] == 1 and calls["cognify"] == 1  # changed/new still re-ingested


def test_no_prune_flag_deletes_nothing():
    _result, deleted, calls = _run_update(
        tmp_files=["present.txt"],
        recorded=[_Rec("drop", "gone.txt")],
        prune=False,
    )
    assert deleted == []
    assert calls["add"] == 1 and calls["cognify"] == 1


def test_first_run_without_dataset_skips_prune():
    result, deleted, _calls = _run_update(
        tmp_files=["present.txt"],
        recorded=[_Rec("drop", "gone.txt")],
        dataset_exists=False,
    )
    assert deleted == []
    assert result["removed"] == 0


def test_out_of_scope_record_is_not_pruned():
    # A recorded row whose absolute location is outside the synced root must survive,
    # even though it's absent from the root's current files.
    outside = os.path.join(os.path.abspath(os.sep), "definitely", "outside", "x.txt")
    result, deleted, _calls = _run_update(
        tmp_files=["present.txt"],
        recorded=[_Rec("outside", outside)],
    )
    assert deleted == []
    assert result["removed"] == 0


def test_missing_root_dir_never_prunes():
    # Safety: a root that is NOT an existing directory (typo, unmounted drive, wrong
    # cwd) must never be read as "everything was deleted" and wipe the dataset — even
    # though its recorded rows are absent from disk.
    import tempfile

    missing = os.path.join(tempfile.gettempdir(), "iu_missing_" + os.urandom(4).hex())
    assert not os.path.exists(missing)

    deleted = []

    async def _setup():
        pass

    async def _add(**kwargs):
        pass

    async def _cognify(**kwargs):
        return {}

    async def _get_default_user():
        return _User()

    async def _get_authorized(datasets, permission_type, user):
        return [_Dataset()]

    async def _get_dataset_data(dataset_id):
        return [_Rec("drop", os.path.join(missing, "gone.txt"))]

    async def _has_nodes(dataset_id, data_id):
        return True

    async def _delete_nodes(dataset_id, data_id, user_id):
        pass

    async def _delete_data(record, dataset_id):
        deleted.append(record.id)

    restore = _patch(
        {
            "setup": _setup,
            "add": _add,
            "cognify": _cognify,
            "get_default_user": _get_default_user,
            "get_authorized_existing_datasets": _get_authorized,
            "get_dataset_data": _get_dataset_data,
            "has_data_related_nodes": _has_nodes,
            "delete_data_nodes_and_edges": _delete_nodes,
            "delete_data": _delete_data,
        }
    )
    try:
        result = _run(iu.incremental_update(missing, dataset_name="ds", prune_removed=True))
    finally:
        restore()
    assert deleted == []
    assert result["removed"] == 0


# --------------------------------------------------------------------------- #
# cognee hook install / uninstall
# --------------------------------------------------------------------------- #
def _in_temp_git_repo(fn):
    import tempfile

    repo = tempfile.mkdtemp(prefix="iu_git_")
    (Path(repo) / ".git" / "hooks").mkdir(parents=True)
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        return fn(Path(repo))
    finally:
        os.chdir(cwd)


def test_hook_install_writes_managed_block_then_uninstall_removes_it():
    def body(repo):
        cmd = HookCommand()
        hook = repo / ".git" / "hooks" / "post-commit"

        cmd.execute(argparse.Namespace(action="install", path=".", dataset_name="proj"))
        text = hook.read_text(encoding="utf-8")
        assert _MARK_START in text and _MARK_END in text
        assert "cognee update" in text and 'proj' in text

        # Idempotent: installing again keeps exactly one managed block.
        cmd.execute(argparse.Namespace(action="install", path=".", dataset_name="proj"))
        assert hook.read_text(encoding="utf-8").count(_MARK_START) == 1

        cmd.execute(argparse.Namespace(action="uninstall", path=".", dataset_name="proj"))
        assert (not hook.exists()) or (_MARK_START not in hook.read_text(encoding="utf-8"))

    _in_temp_git_repo(body)


def test_hook_install_preserves_existing_hook():
    def body(repo):
        hook = repo / ".git" / "hooks" / "post-commit"
        hook.write_text("#!/bin/sh\necho existing-hook\n", encoding="utf-8")

        HookCommand().execute(
            argparse.Namespace(action="install", path=".", dataset_name="main_dataset")
        )
        text = hook.read_text(encoding="utf-8")
        assert "echo existing-hook" in text  # user's hook preserved
        assert _MARK_START in text  # managed block appended

        HookCommand().execute(
            argparse.Namespace(action="uninstall", path=".", dataset_name="main_dataset")
        )
        text = hook.read_text(encoding="utf-8")
        assert "echo existing-hook" in text and _MARK_START not in text

    _in_temp_git_repo(body)


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except AssertionError as exc:
                failures += 1
                print("FAIL", _name, exc)
    raise SystemExit(1 if failures else 0)
