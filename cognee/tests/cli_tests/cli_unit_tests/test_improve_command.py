"""cognee-cli improve: option parity with the SDK and one printed line per stage."""

import argparse
import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.cli.commands.improve_command import (
    ImproveCommand,
    format_stage_line,
    print_improve_result,
)
from cognee.cli.exceptions import CliCommandException
from cognee.modules.improve import ImproveResult, StageResult


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse(*argv):
    parser = argparse.ArgumentParser()
    ImproveCommand().configure_parser(parser)
    return parser.parse_args(list(argv))


def _result(*stages, **fields):
    return ImproveResult(dataset_name="docs", stages=list(stages), memify_run={}, **fields)


class TestParser:
    def test_new_flags_default_off_and_alpha_defaults_to_config(self):
        args = _parse()
        assert args.build_global_context_index is False
        assert args.build_truth_subspace is False
        assert args.feedback_alpha is None
        assert args.background is False
        assert args.dataset_name == "main_dataset"

    def test_flags_are_parsed(self):
        args = _parse(
            "-d",
            "docs",
            "-s",
            "s1",
            "s2",
            "--node-name",
            "Alice",
            "--feedback-alpha",
            "0.3",
            "--build-global-context-index",
            "--build-truth-subspace",
            "-b",
        )
        assert args.session_ids == ["s1", "s2"]
        assert args.node_name == ["Alice"]
        assert args.feedback_alpha == 0.3
        assert args.build_global_context_index is True
        assert args.build_truth_subspace is True
        assert args.background is True


class TestExecute:
    def _execute(self, args, result):
        improve = AsyncMock(return_value=result)
        echoed = []
        with (
            patch("cognee.improve", improve),
            patch("cognee.cli.commands.improve_command.asyncio.run", side_effect=_run),
            patch("cognee.cli.commands.improve_command.fmt.echo", side_effect=echoed.append),
            patch("cognee.cli.commands.improve_command.fmt.success", side_effect=echoed.append),
            patch("cognee.cli.commands.improve_command.fmt.warning", side_effect=echoed.append),
        ):
            ImproveCommand().execute(args)
        return improve, echoed

    def test_forwards_every_option(self):
        dataset_id = uuid4()
        args = _parse(
            "--dataset-id",
            str(dataset_id),
            "-s",
            "s1",
            "--feedback-alpha",
            "0.2",
            "--build-global-context-index",
            "--build-truth-subspace",
        )
        improve, _ = self._execute(args, _result(StageResult.completed("triplet_enrichment")))

        kwargs = improve.call_args.kwargs
        assert kwargs["dataset"] == dataset_id
        assert kwargs["session_ids"] == ["s1"]
        assert kwargs["feedback_alpha"] == 0.2
        assert kwargs["build_global_context_index"] is True
        assert kwargs["build_truth_subspace"] is True
        assert kwargs["run_in_background"] is False

    def test_omitted_alpha_is_not_forwarded(self):
        improve, _ = self._execute(_parse(), _result(StageResult.completed("triplet_enrichment")))
        assert "feedback_alpha" not in improve.call_args.kwargs

    def test_prints_one_line_per_stage(self):
        result = _result(
            StageResult.skipped("feedback_weights", "no_session_ids"),
            StageResult.completed("triplet_enrichment", nodes=12),
            StageResult.errored("global_context_index", RuntimeError("boom")),
        )
        _, echoed = self._execute(_parse(), result)

        text = "\n".join(echoed)
        assert "finished with errors" in text
        assert "feedback_weights" in text and "no_session_ids" in text
        assert "triplet_enrichment" in text and "nodes=12" in text
        assert "global_context_index" in text and "boom" in text
        stage_lines = [line for line in echoed if line.startswith("  ")]
        assert len(stage_lines) == 3

    def test_background_prints_only_the_headline(self):
        result = ImproveResult(dataset_name="docs", background=True)
        result.finished = False
        _, echoed = self._execute(_parse("-b"), result)

        # The "Improving ..." preamble, then the headline — no stage lines.
        assert echoed[-1] == "Improvement started in background!"
        assert len(echoed) == 2

    def test_lost_lock_is_explained(self):
        result = ImproveResult.all_skipped(["feedback_weights", "triplet_enrichment"], "lock_held")
        _, echoed = self._execute(_parse(), result)

        assert any("lock_held" in line for line in echoed)
        assert not any("successfully" in line for line in echoed)

    def test_failure_becomes_a_cli_error(self):
        with (
            patch("cognee.improve", AsyncMock(side_effect=RuntimeError("nope"))),
            patch("cognee.cli.commands.improve_command.asyncio.run", side_effect=_run),
            patch("cognee.cli.commands.improve_command.fmt.echo"),
            pytest.raises(CliCommandException, match="Failed to improve: nope"),
        ):
            ImproveCommand().execute(_parse())


class TestFormatting:
    def test_stage_line_from_model_and_from_dict_agree(self):
        stage = StageResult.skipped("build_truth_subspace", "opt_in_disabled")
        assert format_stage_line(stage) == format_stage_line(stage.model_dump(mode="json"))
        assert "build_truth_subspace" in format_stage_line(stage)
        assert "opt_in_disabled" in format_stage_line(stage)

    def test_stage_line_carries_counts_and_duration(self):
        stage = StageResult.completed("distill_sessions", lessons=4)
        stage.duration_ms = 250
        line = format_stage_line(stage)
        assert "completed" in line and "lessons=4" in line and "250 ms" in line

    def test_print_accepts_the_serialized_api_payload(self):
        payload = _result(
            StageResult.completed("triplet_enrichment"),
            StageResult.skipped("global_context_index", "opt_in_disabled"),
        ).model_dump(mode="json")
        echoed = []
        with (
            patch("cognee.cli.commands.improve_command.fmt.echo", side_effect=echoed.append),
            patch("cognee.cli.commands.improve_command.fmt.success", side_effect=echoed.append),
        ):
            print_improve_result(payload)

        assert echoed[0] == "Knowledge graph improved successfully!"
        assert len(echoed) == 3
        assert "opt_in_disabled" in echoed[2]
