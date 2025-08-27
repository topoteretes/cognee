"""
Integration tests for CLI commands that test end-to-end functionality.
"""

import tempfile
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCliIntegration:
    """Integration tests for CLI commands"""

    def test_cli_help(self):
        """Test that CLI help works"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode == 0
        assert "cognee" in result.stdout.lower()
        assert "available commands" in result.stdout.lower()

    def test_cli_version(self):
        """Test that CLI version works"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode == 0
        assert "cognee" in result.stdout.lower()

    def test_command_help(self):
        """Test that individual command help works"""
        commands = ["add", "search", "cognify", "delete", "config"]

        for command in commands:
            result = subprocess.run(
                [sys.executable, "-m", "cognee.cli._cognee", command, "--help"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Go to project root
            )

            assert result.returncode == 0, f"Command {command} help failed"
            assert command in result.stdout.lower()

    def test_invalid_command(self):
        """Test that invalid commands are handled properly"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "invalid_command"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode != 0

    @patch("cognee.add")
    def test_add_command_integration(self, mock_add):
        """Test add command integration"""
        mock_add.return_value = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for CLI integration")
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "cognee.cli._cognee", "add", temp_file],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Go to project root
            )

            # Note: This might fail due to dependencies, but we're testing the CLI structure
            # The important thing is that it doesn't crash with argument parsing errors
            # Allow litellm logging worker cancellation errors as they're expected during process shutdown
            stderr_lower = result.stderr.lower()
            has_error = "error" in stderr_lower
            has_expected_failure = "failed to add data" in stderr_lower
            has_litellm_cancellation = (
                "loggingworker cancelled" in stderr_lower or "cancellederror" in stderr_lower
            )

            assert not has_error or has_expected_failure or has_litellm_cancellation

        finally:
            os.unlink(temp_file)

    def test_config_subcommands(self):
        """Test config subcommands help"""
        subcommands = ["get", "set", "list", "unset", "reset"]

        for subcommand in subcommands:
            result = subprocess.run(
                [sys.executable, "-m", "cognee.cli._cognee", "config", subcommand, "--help"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Go to project root
            )

            assert result.returncode == 0, f"Config {subcommand} help failed"

    def test_search_command_missing_query(self):
        """Test search command fails when query is missing"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "search"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_delete_command_no_target(self):
        """Test delete command with no target specified"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "delete"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        # Should run but show error message about missing target
        # Return code might be 0 since the command handles this gracefully
        assert (
            "specify what to delete" in result.stdout.lower()
            or "specify what to delete" in result.stderr.lower()
        )


class TestCliArgumentParsing:
    """Test CLI argument parsing edge cases"""

    def test_add_multiple_files(self):
        """Test add command with multiple file arguments"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = os.path.join(temp_dir, "file1.txt")
            file2 = os.path.join(temp_dir, "file2.txt")

            with open(file1, "w") as f:
                f.write("Content 1")
            with open(file2, "w") as f:
                f.write("Content 2")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cognee.cli._cognee",
                    "add",
                    file1,
                    file2,
                    "--dataset-name",
                    "test",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,  # Go to project root
            )

            # Test that argument parsing works (regardless of actual execution)
            assert (
                "argument" not in result.stderr.lower() or "failed to add" in result.stderr.lower()
            )

    def test_search_with_all_options(self):
        """Test search command with all possible options"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cognee.cli._cognee",
                "search",
                "test query",
                "--query-type",
                "CHUNKS",
                "--datasets",
                "dataset1",
                "dataset2",
                "--top-k",
                "5",
                "--output-format",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        # Should not have argument parsing errors
        assert "unrecognized arguments" not in result.stderr.lower()
        assert "invalid choice" not in result.stderr.lower()

    def test_cognify_with_all_options(self):
        """Test cognify command with all possible options"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cognee.cli._cognee",
                "cognify",
                "--datasets",
                "dataset1",
                "dataset2",
                "--chunk-size",
                "1024",
                "--chunker",
                "TextChunker",
                "--background",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        # Should not have argument parsing errors
        assert "unrecognized arguments" not in result.stderr.lower()
        assert "invalid choice" not in result.stderr.lower()

    def test_config_set_command(self):
        """Test config set command argument parsing"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "config", "set", "test_key", "test_value"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        # Should not have argument parsing errors
        assert "unrecognized arguments" not in result.stderr.lower()
        assert "required" not in result.stderr.lower() or "failed to set" in result.stderr.lower()

    def test_delete_with_force(self):
        """Test delete command with force flag"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cognee.cli._cognee",
                "delete",
                "--dataset-name",
                "test_dataset",
                "--force",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        # Should not have argument parsing errors
        assert "unrecognized arguments" not in result.stderr.lower()


class TestCliErrorHandling:
    """Test CLI error handling and edge cases"""

    def test_debug_mode_flag(self):
        """Test that debug flag is accepted"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "--debug", "search", "test query"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        # Should not have argument parsing errors for debug flag
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_invalid_search_type(self):
        """Test invalid search type handling"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cognee.cli._cognee",
                "search",
                "test query",
                "--query-type",
                "INVALID_TYPE",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_invalid_chunker(self):
        """Test invalid chunker handling"""
        result = subprocess.run(
            [sys.executable, "-m", "cognee.cli._cognee", "cognify", "--chunker", "InvalidChunker"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_invalid_output_format(self):
        """Test invalid output format handling"""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cognee.cli._cognee",
                "search",
                "test query",
                "--output-format",
                "invalid",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Go to project root
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()
