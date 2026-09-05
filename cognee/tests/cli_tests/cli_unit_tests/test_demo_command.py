"""Tests for the demo CLI command and its bundled COGX archive."""

import argparse
import asyncio
import json
from importlib import resources
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.cli.commands.demo_command import DemoCommand, _result_lines
from cognee.cli.exceptions import CliCommandException


def _mock_run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _remember_result(nodes=37, edges=71):
    result = MagicMock()
    result.items = [{"graph_nodes": nodes, "graph_edges": edges}]
    return result


def _parse_args(command, argv):
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    return parser.parse_args(argv)


class TestDemoCommand:
    def test_command_properties(self):
        command = DemoCommand()
        assert command.command_string == "demo"
        assert "no API key" in command.help_string
        assert command.docs_url is not None

    def test_parser_defaults(self):
        args = _parse_args(DemoCommand(), [])
        assert args.dataset_name == "demo"
        assert args.query is None
        assert args.top_k == 3

    def test_execute_runs_graph_only_import_and_lexical_search(self):
        import cognee
        from cognee.modules.search.types import SearchType

        remember_mock = AsyncMock(return_value=_remember_result())
        search_mock = AsyncMock(return_value=[{"text": "Alice works at Anthropic."}])

        with (
            patch.object(cognee, "remember", remember_mock),
            patch.object(cognee, "search", search_mock),
            patch("asyncio.run", _mock_run),
        ):
            DemoCommand().execute(_parse_args(DemoCommand(), []))

        # The import must be graph-only (no embeddings -> no API key needed).
        assert remember_mock.await_count == 1
        _, remember_kwargs = remember_mock.await_args
        assert remember_kwargs["index_vectors"] is False
        assert remember_kwargs["dataset_name"] == "demo"
        source = remember_mock.await_args.args[0]
        assert type(source).__name__ == "COGXArchiveSource"
        assert source.mode == "preserve"

        # Every search must be lexical (the only zero-LLM, zero-embedding type).
        assert search_mock.await_count >= 1
        for call in search_mock.await_args_list:
            assert call.kwargs["query_type"] == SearchType.CHUNKS_LEXICAL
            assert call.kwargs["datasets"] == ["demo"]

    def test_execute_custom_query_and_dataset(self):
        import cognee

        remember_mock = AsyncMock(return_value=_remember_result())
        search_mock = AsyncMock(return_value=[])

        with (
            patch.object(cognee, "remember", remember_mock),
            patch.object(cognee, "search", search_mock),
            patch("asyncio.run", _mock_run),
        ):
            DemoCommand().execute(
                _parse_args(DemoCommand(), ["-q", "Where does Alice live?", "-d", "my_demo"])
            )

        assert search_mock.await_count == 1
        assert search_mock.await_args.kwargs["query_text"] == "Where does Alice live?"
        assert remember_mock.await_args.kwargs["dataset_name"] == "my_demo"

    def test_execute_wraps_failures_in_cli_exception(self):
        import cognee

        with (
            patch.object(cognee, "remember", AsyncMock(side_effect=RuntimeError("boom"))),
            patch("asyncio.run", _mock_run),
        ):
            with pytest.raises(CliCommandException, match="boom"):
                DemoCommand().execute(_parse_args(DemoCommand(), []))


class TestResultLines:
    def test_flattens_nested_payloads(self):
        results = [
            [{"text": "first chunk"}],
            {"search_result": [{"text": "second chunk"}]},
            "bare string",
        ]
        lines = _result_lines(results)
        assert "first chunk" in lines
        assert "second chunk" in lines
        assert "bare string" in lines

    def test_empty_results(self):
        assert _result_lines([]) == []
        assert _result_lines(None) == []


class TestBundledArchive:
    """The committed demo archive must stay importable and key-free."""

    def _archive_dir(self):
        return resources.files("cognee.cli.samples").joinpath("demo_graph")

    def test_archive_ships_with_the_package(self):
        archive = self._archive_dir()
        assert archive.joinpath("manifest.json").is_file()
        assert archive.joinpath("nodes.jsonl").is_file()
        assert archive.joinpath("facts.jsonl").is_file()

    def test_archive_carries_no_document_records(self):
        """Document records would make the import call add(), whose pipeline
        demands a working LLM connection — the demo must never need one."""
        archive = self._archive_dir()
        assert not archive.joinpath("documents.jsonl").is_file()
        manifest = json.loads(archive.joinpath("manifest.json").read_text(encoding="utf-8"))
        assert "document" not in manifest.get("counts", {})

    def test_archive_parses_into_graph_records_only(self):
        from cognee.modules.migration.cogx import read_archive

        kinds = {record.kind for record in read_archive(str(self._archive_dir()))}
        assert kinds <= {"entity", "fact", "raw_node"}
        assert "fact" in kinds and "raw_node" in kinds

    def test_raw_nodes_include_lexical_searchable_chunks(self):
        """CHUNKS_LEXICAL reads DocumentChunk nodes from the graph; the export
        writes each chunk as a raw node, and the demo relies on that."""
        archive = self._archive_dir()
        chunk_types = set()
        for line in archive.joinpath("nodes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                properties = json.loads(line)
                chunk_types.add(properties.get("type"))
        assert "DocumentChunk" in chunk_types

    def _chunk_texts(self):
        archive = self._archive_dir()
        chunks = []
        for line in archive.joinpath("nodes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                node = json.loads(line)
                if node.get("type") == "DocumentChunk":
                    chunks.append(node.get("text") or "")
        return chunks

    def test_chunks_are_displayable_answers(self):
        """The demo prints whole chunks as answers (no truncation), so the
        archive must ship several small chunks with each built-in query's
        fact in its own chunk — a single document-sized chunk made every
        query print the same answer-free intro paragraph."""
        chunks = self._chunk_texts()
        assert len(chunks) >= 3
        assert any("Alice works at Anthropic" in chunk for chunk in chunks)
        assert any("depends on litellm" in chunk for chunk in chunks)
        # Printed in full, every chunk must stay terminal-sized.
        assert all(len(chunk) <= 400 for chunk in chunks)

    def test_archive_leaks_no_build_machine_paths(self):
        """The builder nulls raw_data_location: the bundled archive must not
        ship the maintainer's filesystem layout or dead file:// provenance."""
        archive = self._archive_dir()
        for line in archive.joinpath("nodes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                node = json.loads(line)
                assert not node.get("raw_data_location")
