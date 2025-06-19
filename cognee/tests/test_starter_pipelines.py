import unittest
import subprocess
import os
import sys


class TestPipelines(unittest.TestCase):
    """Tests that all pipelines run successfully."""

    def setUp(self):
        # Ensure we're in the correct directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.pipelines_dir = os.path.join(self.project_root, "src", "pipelines")

        # Required environment variables
        self.required_env_vars = ["LLM_API_KEY", "EMBEDDING_API_KEY"]

        # Check if required environment variables are set
        missing_vars = [var for var in self.required_env_vars if not os.environ.get(var)]
        if missing_vars:
            self.skipTest(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _run_pipeline(self, script_name):
        """Helper method to run a pipeline script and return the result."""
        script_path = os.path.join(self.pipelines_dir, script_name)

        # Use the Python executable from the virtual environment
        python_exe = os.path.join(self.project_root, ".venv", "bin", "python")
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        try:
            result = subprocess.run(
                [python_exe, script_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return result
        except subprocess.CalledProcessError as e:
            self.fail(
                f"Pipeline {script_name} failed with code {e.returncode}. "
                f"Stdout: {e.stdout}, Stderr: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            self.fail(f"Pipeline {script_name} timed out after 300 seconds")

    def test_default_pipeline(self):
        """Test that the default pipeline runs successfully."""
        result = self._run_pipeline("default.py")
        self.assertEqual(result.returncode, 0)

    def test_low_level_pipeline(self):
        """Test that the low-level pipeline runs successfully."""
        result = self._run_pipeline("low_level.py")
        self.assertEqual(result.returncode, 0)

    def test_custom_model_pipeline(self):
        """Test that the custom model pipeline runs successfully."""
        result = self._run_pipeline("custom-model.py")
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
