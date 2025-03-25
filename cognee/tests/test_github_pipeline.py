"""
Test for GitHub developer analysis pipeline.
"""
import asyncio
import logging
import unittest
from uuid import uuid4
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognee.api.v1.cognify.github_developer_pipeline import run_github_developer_pipeline
from cognee.modules.data.deletion import prune_data, prune_system


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestGitHubPipeline(unittest.TestCase):
    """Test the GitHub developer analysis pipeline."""

    def setUp(self):
        """Set up test case."""
        self.github_username = "Vasilije1990"
        self.api_token = os.environ.get("GITHUB_API_TOKEN")  # Optional, can be None
        
    async def async_test_pipeline(self):
        """Run the GitHub pipeline and check results."""
        logger.info(f"Testing GitHub pipeline for user: {self.github_username}")
        
        # Prune system first to clear previous data
        logger.info("Pruning data and system...")
        await prune_data()
        await prune_system(metadata=True)
        
        # Setup the pipeline
        logger.info("Setting up and running GitHub pipeline...")
        
        # Run the pipeline
        steps_completed = 0
        async for status in run_github_developer_pipeline(self.github_username, self.api_token):
            logger.info(f"Pipeline status: {status}")
            steps_completed += 1
        
        # Verify that the pipeline completed successfully
        logger.info(f"Pipeline completed {steps_completed} steps")
        
        # Here we just check that some steps were completed
        self.assertGreater(steps_completed, 0, "Pipeline did not complete any steps")

    def test_github_pipeline(self):
        """Test wrapper to run async test."""
        asyncio.run(self.async_test_pipeline())


if __name__ == "__main__":
    unittest.main() 