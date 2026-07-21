import argparse
from types import SimpleNamespace

import pytest

from cognee.cli.commands._next_step import format_next_step_hint

add_module = pytest.importorskip("cognee.cli.commands.add_command")
cognify_module = pytest.importorskip("cognee.cli.commands.cognify_command")
remember_module = pytest.importorskip("cognee.cli.commands.remember_command")
recall_module = pytest.importorskip("cognee.cli.commands.recall_command")


def _capture_cli(monkeypatch):
    captured = []

    monkeypatch.setattr(add_module.fmt, "echo", lambda message: captured.append(message))
    monkeypatch.setattr(add_module.fmt, "success", lambda message: captured.append(message))
    monkeypatch.setattr(add_module.fmt, "note", lambda message: captured.append(message))
    monkeypatch.setattr(add_module.fmt, "warning", lambda message: captured.append(message))

    monkeypatch.setattr(cognify_module.fmt, "echo", lambda message: captured.append(message))
    monkeypatch.setattr(cognify_module.fmt, "success", lambda message: captured.append(message))
    monkeypatch.setattr(cognify_module.fmt, "note", lambda message: captured.append(message))
    monkeypatch.setattr(cognify_module.fmt, "warning", lambda message: captured.append(message))

    monkeypatch.setattr(remember_module.fmt, "echo", lambda message: captured.append(message))
    monkeypatch.setattr(remember_module.fmt, "success", lambda message: captured.append(message))
    monkeypatch.setattr(remember_module.fmt, "note", lambda message: captured.append(message))
    monkeypatch.setattr(remember_module.fmt, "warning", lambda message: captured.append(message))

    monkeypatch.setattr(recall_module.fmt, "echo", lambda message: captured.append(message))
    monkeypatch.setattr(recall_module.fmt, "success", lambda message: captured.append(message))
    monkeypatch.setattr(recall_module.fmt, "note", lambda message: captured.append(message))
    monkeypatch.setattr(recall_module.fmt, "warning", lambda message: captured.append(message))

    return captured


def test_format_next_step_hint_uses_dataset_name():
    assert format_next_step_hint("add", "main_dataset") == "Next: `cognee cognify --datasets main_dataset`"
    assert format_next_step_hint("recall", "main_dataset") == (
        "Try adding data first: `cognee remember \"...\" --dataset-name main_dataset`"
    )


def test_add_command_prints_next_step_hint(monkeypatch):
    captured = _capture_cli(monkeypatch)

    async def fake_add(**kwargs):
        return None

    async def fake_resolve_cli_user(_user_id, strict=False):
        return SimpleNamespace(id="user-1")

    import cognee

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", fake_resolve_cli_user)

    command = add_module.AddCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    args = parser.parse_args(["hello", "--dataset-name", "main_dataset"])

    command.execute(args)

    assert any("Successfully added data" in message for message in captured)
    assert any("Next: `cognee cognify --datasets main_dataset`" in message for message in captured)


def test_cognify_command_prints_next_step_hint(monkeypatch):
    captured = _capture_cli(monkeypatch)

    async def fake_cognify(**kwargs):
        return True

    async def fake_resolve_cli_user(_user_id, strict=False):
        return SimpleNamespace(id="user-1")

    class FakeTextChunker:
        pass

    import cognee

    monkeypatch.setattr(cognee, "cognify", fake_cognify)
    monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", fake_resolve_cli_user)
    monkeypatch.setattr("cognee.modules.chunking.TextChunker.TextChunker", FakeTextChunker, raising=False)

    command = cognify_module.CognifyCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    args = parser.parse_args(["--datasets", "main_dataset"])

    command.execute(args)

    assert any("Cognification completed successfully!" in message for message in captured)
    assert any("Next: `cognee recall \"What did I add?\" --datasets main_dataset`" in message for message in captured)


def test_remember_command_prints_next_step_hint(monkeypatch):
    captured = _capture_cli(monkeypatch)

    async def fake_remember(**kwargs):
        return SimpleNamespace(
            dataset_id="dataset-1",
            items_processed=1,
            content_hash="hash",
            elapsed_seconds=1.2,
        )

    async def fake_resolve_cli_user(_user_id, strict=False):
        return SimpleNamespace(id="user-1")

    monkeypatch.setattr("cognee.modules.chunking.TextChunker.TextChunker", object, raising=False)
    monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", fake_resolve_cli_user)
    import cognee

    monkeypatch.setattr(cognee, "remember", fake_remember)

    command = remember_module.RememberCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    args = parser.parse_args(["hello", "--dataset-name", "main_dataset"])

    command.execute(args)

    assert any("Data ingested and knowledge graph built successfully!" in message for message in captured)
    assert any("Next: `cognee recall \"What did I add?\" --datasets main_dataset`" in message for message in captured)


def test_recall_command_prints_next_step_hint_when_empty(monkeypatch):
    captured = _capture_cli(monkeypatch)

    async def fake_recall(**kwargs):
        return []

    async def fake_resolve_cli_user(_user_id, strict=False):
        return SimpleNamespace(id="user-1")

    monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", fake_resolve_cli_user)
    import cognee

    monkeypatch.setattr(cognee, "recall", fake_recall)

    command = recall_module.RecallCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    args = parser.parse_args(["hello", "--datasets", "main_dataset"])

    command.execute(args)

    assert any("No results found for your query." in message for message in captured)
    assert any("Try adding data first: `cognee remember \"...\" --dataset-name main_dataset`" in message for message in captured)
