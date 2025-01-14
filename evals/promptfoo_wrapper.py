import subprocess
import json
import logging
import os
from typing import List, Optional, Dict, Generator
import shutil
import platform
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class PromptfooWrapper:
    """
    A Python wrapper class around the promptfoo CLI tool, allowing you to:
    - Evaluate prompts against different language models.
    - Compare responses from multiple models.
    - Pass configuration and prompt files.
    - Retrieve the outputs in a structured format, including binary output if needed.

    This class assumes you have the promptfoo CLI installed and accessible in your environment.
    For more details on promptfoo, see: https://github.com/promptfoo/promptfoo
    """

    def __init__(self, promptfoo_path: str = ""):
        """
        Initialize the wrapper with the path to the promptfoo executable.

        :param promptfoo_path: Path to the promptfoo binary (default: 'promptfoo')
        """
        self.promptfoo_path = promptfoo_path
        logger.debug(f"Initialized PromptfooWrapper with binary at: {self.promptfoo_path}")

    def _validate_path(self, file_path: Optional[str]) -> None:
        """
        Validate that a file path is accessible if provided.
        Raise FileNotFoundError if it does not exist.
        """
        if file_path and not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    def _get_node_bin_dir(self) -> str:
        """
        Determine the Node.js binary directory dynamically for macOS and Linux.
        """
        node_executable = shutil.which("node")
        if not node_executable:
            logger.error("Node.js is not installed or not found in the system PATH.")
            raise EnvironmentError("Node.js is not installed or not in PATH.")

        # Determine the Node.js binary directory
        node_bin_dir = os.path.dirname(node_executable)

        # Special handling for macOS, where Homebrew installs Node in /usr/local or /opt/homebrew
        if platform.system() == "Darwin":  # macOS
            logger.debug("Running on macOS")
            brew_prefix = os.popen("brew --prefix node").read().strip()
            if brew_prefix and os.path.exists(brew_prefix):
                node_bin_dir = os.path.join(brew_prefix, "bin")
                logger.debug(f"Detected Node.js binary directory using Homebrew: {node_bin_dir}")

        # For Linux, Node.js installed via package managers should work out of the box
        logger.debug(f"Detected Node.js binary directory: {node_bin_dir}")
        return node_bin_dir

    def _run_command(
        self,
        cmd: List[str],
        filename,
    ) -> Generator[Dict, None, None]:
        """
        Run a given command using subprocess and parse the output.
        """
        logger.debug(f"Running command: {' '.join(cmd)}")

        # Make a copy of the current environment
        env = os.environ.copy()

        try:
            node_bin_dir = self._get_node_bin_dir()
            print(node_bin_dir)
            env["PATH"] = f"{node_bin_dir}:{env['PATH']}"

        except EnvironmentError as e:
            logger.error(f"Failed to set Node.js binary directory: {e}")
            raise

        # Add node's bin directory to the PATH
        # node_bin_dir = "/Users/vasilije/Library/Application Support/JetBrains/PyCharm2024.2/node/versions/20.15.0/bin"
        # # env["PATH"] = f"{node_bin_dir}:{env['PATH']}"

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

        print(result.stderr)
        with open(filename, "r", encoding="utf-8") as file:
            read_data = json.load(file)
        print(f"{filename} created and written.")

        # Log raw stdout for debugging
        logger.debug(f"Raw command output:\n{result.stdout}")

        # Use the parse_promptfoo_output function to yield parsed results
        return read_data

    def run_eval(
        self,
        prompt_file: Optional[str] = None,
        config_file: Optional[str] = None,
        eval_file: Optional[str] = None,
        out_format: str = "json",
        extra_args: Optional[List[str]] = None,
        binary_output: bool = False,
    ) -> Dict:
        """
        Run the `promptfoo eval` command with the provided parameters and return parsed results.

        :param prompt_file: Path to a file containing one or more prompts.
        :param config_file: Path to a config file specifying models, scoring methods, etc.
        :param eval_file: Path to an eval file with test data.
        :param out_format: Output format, e.g., 'json', 'yaml', or 'table'.
        :param extra_args: Additional command-line arguments for fine-tuning evaluation.
        :param binary_output: If True, interpret output as binary data instead of text.
        :return: List of parsed results (each result is a dictionary).
        """
        self._validate_path(prompt_file)
        self._validate_path(config_file)
        self._validate_path(eval_file)

        filename = "benchmark_results"

        filename = os.path.join(os.getcwd(), f"{filename}.json")
        # Create an empty JSON file
        with open(filename, "w") as file:
            json.dump({}, file)

        cmd = [self.promptfoo_path, "eval"]
        if prompt_file:
            cmd.extend(["--prompts", prompt_file])
        if config_file:
            cmd.extend(["--config", config_file])
        if eval_file:
            cmd.extend(["--eval", eval_file])
        cmd.extend(["--output", filename])
        if extra_args:
            cmd.extend(extra_args)

        # Log the constructed command for debugging
        logger.debug(f"Constructed command: {' '.join(cmd)}")

        # Collect results from the generator
        results = self._run_command(cmd, filename=filename)
        logger.debug(f"Parsed results: {json.dumps(results, indent=4)}")
        return results
