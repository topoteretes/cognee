"""
Run the GitHub developer analysis pipeline directly.

This script prunes the system and runs the GitHub developer pipeline
for a specified GitHub username.

The pipeline is limited to:
- Up to 5 repositories per developer (most recently updated)
- Up to 10 contributors per repository
- Up to 5 pull requests per repository
- Up to 20 comments/discussions per pull request

Usage:
    python run_github_analysis.py [github_username] [--tasks task1,task2,...]
"""
import os
import asyncio
import logging
from typing import Optional, List
import argparse
import sys
from pathlib import Path

# Add project root to path
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))

from cognee.api.v1.prune.prune import prune
from cognee.api.v1.cognify.github_developer_pipeline import run_github_developer_pipeline
from cognee.modules.visualization.cognee_network_visualization import cognee_network_visualization
from cognee.infrastructure.databases.graph import get_graph_engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def run_github_analysis(
    username: str,
    full: bool = False,
    tasks: Optional[List[str]] = None,
    repo_limit: int = 5,
    contributor_limit: int = 10,
    pr_limit: int = 5,
    comment_limit: int = 20,
):
    """
    Run the GitHub developer analysis pipeline for a given username.
    
    Args:
        username (str): GitHub username to analyze
        full (bool, optional): Run full analysis (ignores limits). Defaults to False.
        tasks (List[str], optional): Specific tasks to run. Defaults to None.
        repo_limit (int, optional): Limit on repositories to fetch. Defaults to 5.
        contributor_limit (int, optional): Limit on contributors to fetch per repo. Defaults to 10.
        pr_limit (int, optional): Limit on PRs to fetch per repo. Defaults to 5.
        comment_limit (int, optional): Limit on comments to fetch per PR. Defaults to 20.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing GitHub user: {username}")
    
    # Log which tasks will run
    if tasks:
        logger.info(f"Running specific tasks: {', '.join(tasks)}")
    
    # Log limits for better performance
    logger.info(f"Using data limits for better performance: max {repo_limit} repos, {contributor_limit} contributors per repo, {pr_limit} PRs per repo, {comment_limit} comments per PR")
    
    # Create an instance of OpenAIAdapter for LLM
    token = os.environ.get("OPENAI_API_KEY")
    if not token:
        logger.warning("No OpenAI API key found in environment variables. Set the OPENAI_API_KEY environment variable.")
        return
    
    logger.info("Using GitHub API token from environment/settings")
    
    # Prune data
    logger.info("Pruning data and system...")
    await prune.prune_data()

    # Get GitHub API token
    github_token = os.environ.get("GITHUB_API_TOKEN")

    # Run the GitHub analysis pipeline
    logger.info(f"Starting GitHub analysis pipeline for {username}...")
    
    # Pass parameters correctly based on the function signature
    async for status in run_github_developer_pipeline(
        github_username=username, 
        api_token=github_token,
        run_tasks_list=tasks
    ):
        logger.info(f"Pipeline status: {status}")
    
    # Visualize the knowledge graph
    logger.info("Visualizing knowledge graph...")
    graph_engine = get_graph_engine()
    
    # Get all nodes and edges from the graph
    nodes = await graph_engine.get_all_nodes()
    edges = await graph_engine.get_all_edges()
    
    # Format graph data correctly for visualization function
    graph_data = {
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges]
    }
    
    # Visualize the graph
    home_dir = os.path.expanduser("~")
    output_path = os.path.join(home_dir, "graph_visualization.html")
    await cognee_network_visualization(graph_data, output_path)
    
    logger.info(f"The HTML file has been stored at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GitHub analysis on a user")
    parser.add_argument("username", help="GitHub username")
    parser.add_argument("--tasks", help="Comma-separated tasks to run", default=None)
    parser.add_argument("--full", help="Run full analysis (ignore limits)", action="store_true")
    parser.add_argument("--repo-limit", help="Max repositories to fetch", type=int, default=5)
    parser.add_argument("--contributor-limit", help="Max contributors per repo", type=int, default=10)
    parser.add_argument("--pr-limit", help="Max PRs per repo", type=int, default=5)
    parser.add_argument("--comment-limit", help="Max comments per PR", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Parse tasks if provided
    task_list = args.tasks.split(",") if args.tasks else None
    
    asyncio.run(run_github_analysis(
        username=args.username,
        full=args.full,
        tasks=task_list,
        repo_limit=args.repo_limit,
        contributor_limit=args.contributor_limit,
        pr_limit=args.pr_limit,
        comment_limit=args.comment_limit
    )) 