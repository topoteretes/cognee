import json
import asyncio
from uuid import uuid5, NAMESPACE_OID
from typing import Optional, List, Dict, Any
import cognee
from cognee.low_level import DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.pipelines import run_tasks
from cognee.modules.users.methods import get_default_user
from cognee.modules.engine.models.node_set import NodeSet
from cognee.shared.logging_utils import get_logger
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_ingest import (
    get_github_data_for_cognee,
)
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunStarted
from cognee.modules.graph.operations import get_formatted_graph_data
from cognee.modules.crewai.get_crewai_pipeline_run_id import get_crewai_pipeline_run_id

# Import DataPoint classes from github_datapoints.py
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_datapoints import (
    GitHubUser,
    Repository,
    File,
    Commit,
)

# Import creator functions from github_datapoint_creators.py
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_datapoint_creators import (
    create_github_user_datapoint,
    create_repository_datapoint,
    create_file_datapoint,
    create_commit_datapoint,
    create_file_change_datapoint,
    create_issue_datapoint,
    create_comment_datapoint,
)

logger = get_logger("github_ingest")


def collect_repositories(
    section: List[Dict[str, Any]],
    repositories: Dict[str, Repository],
    user: GitHubUser,
    nodesets: List[NodeSet],
) -> None:
    """Collect unique repositories from a data section and register them to the user."""
    for entry in section:
        repo_name = entry.get("repo", "")
        if not repo_name or repo_name in repositories:
            continue
        repo = create_repository_datapoint(repo_name, nodesets)
        repositories[repo_name] = repo
        user.interacts_with.append(repo)


def get_or_create_repository(
    repo_name: str, repositories: Dict[str, Repository], user: GitHubUser, nodesets: List[NodeSet]
) -> Repository:
    if repo_name in repositories:
        return repositories[repo_name]
    repo = create_repository_datapoint(repo_name, nodesets)
    repositories[repo_name] = repo
    user.interacts_with.append(repo)
    return repo


def get_or_create_file(
    filename: str,
    repo_name: str,
    files: Dict[str, File],
    technical_nodeset: NodeSet,
) -> File:
    file_key = f"{repo_name}:{filename}"
    if file_key in files:
        return files[file_key]
    file = create_file_datapoint(filename, repo_name, [technical_nodeset])
    files[file_key] = file
    return file


def get_or_create_commit(
    commit_data: Dict[str, Any],
    user: GitHubUser,
    commits: Dict[str, Commit],
    repository: Repository,
    technical_nodeset: NodeSet,
) -> Commit:
    commit_sha = commit_data.get("commit_sha", "")
    if commit_sha in commits:
        return commits[commit_sha]
    commit = create_commit_datapoint(commit_data, user, [technical_nodeset])
    commits[commit_sha] = commit
    link_commit_to_repo(commit, repository)
    return commit


def link_file_to_repo(file: File, repository: Repository):
    if file not in repository.contains:
        repository.contains.append(file)


def link_commit_to_repo(commit: Commit, repository: Repository):
    if commit not in repository.has_commit:
        repository.has_commit.append(commit)


def process_file_changes_data(
    github_data: Dict[str, Any],
    user: GitHubUser,
    repositories: Dict[str, Repository],
    technical_nodeset: NodeSet,
) -> List[DataPoint]:
    """Process file changes data and build the graph structure with stronger connections."""
    file_changes = github_data.get("file_changes", [])
    if not file_changes:
        return []

    collect_repositories(file_changes, repositories, user, [technical_nodeset])

    files = {}
    commits = {}
    file_changes_list = []
    for fc_data in file_changes:
        repo_name = fc_data.get("repo", "")
        filename = fc_data.get("filename", "")
        commit_sha = fc_data.get("commit_sha", "")
        if not repo_name or not filename or not commit_sha:
            continue
        repository = get_or_create_repository(repo_name, repositories, user, [technical_nodeset])
        file = get_or_create_file(filename, repo_name, files, technical_nodeset)
        commit = get_or_create_commit(fc_data, user, commits, repository, technical_nodeset)
        file_change = create_file_change_datapoint(fc_data, user, file, [technical_nodeset])
        file_changes_list.append(file_change)
        if file_change not in commit.has_change:
            commit.has_change.append(file_change)
    all_datapoints = list(commits.values()) + file_changes_list
    return all_datapoints


def process_comments_data(
    github_data: Dict[str, Any],
    user: GitHubUser,
    repositories: Dict[str, Repository],
    technical_nodeset: NodeSet,
    soft_nodeset: NodeSet,
) -> List[DataPoint]:
    """Process comments data and build the graph structure with stronger connections."""
    comments_data = github_data.get("comments", [])
    if not comments_data:
        return []

    collect_repositories(comments_data, repositories, user, [soft_nodeset])

    issues = {}
    comments_list = []
    for comment_data in comments_data:
        repo_name = comment_data.get("repo", "")
        issue_number = comment_data.get("issue_number", 0)
        if not repo_name or not issue_number:
            continue
        repository = get_or_create_repository(repo_name, repositories, user, [soft_nodeset])
        issue_key = f"{repo_name}:{issue_number}"
        if issue_key not in issues:
            issue = create_issue_datapoint(comment_data, repo_name, [soft_nodeset])
            issues[issue_key] = issue
            if issue not in repository.has_issue:
                repository.has_issue.append(issue)
        comment = create_comment_datapoint(comment_data, user, [soft_nodeset])
        comments_list.append(comment)
        if comment not in issues[issue_key].has_comment:
            issues[issue_key].has_comment.append(comment)
    all_datapoints = list(issues.values()) + comments_list
    return all_datapoints


def build_github_datapoints_from_dict(github_data: Dict[str, Any]):
    """Builds all DataPoints from a GitHub data dictionary."""
    if not github_data or "user" not in github_data:
        return None

    soft_nodeset = NodeSet(id=uuid5(NAMESPACE_OID, "NodeSet:soft"), name="soft")
    technical_nodeset = NodeSet(id=uuid5(NAMESPACE_OID, "NodeSet:technical"), name="technical")

    datapoints = create_github_user_datapoint(
        github_data["user"], [soft_nodeset, technical_nodeset]
    )
    if not datapoints:
        return None
    user = datapoints[0]

    repositories = {}

    file_change_datapoints = process_file_changes_data(
        github_data, user, repositories, technical_nodeset
    )
    comment_datapoints = process_comments_data(
        github_data, user, repositories, technical_nodeset, soft_nodeset
    )

    all_datapoints = (
        datapoints + list(repositories.values()) + file_change_datapoints + comment_datapoints
    )
    return all_datapoints


async def run_with_info_stream(tasks, user, data, dataset_id, pipeline_name):
    from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue

    pipeline_run = run_tasks(
        tasks=tasks,
        data=data,
        dataset_id=dataset_id,
        pipeline_name=pipeline_name,
        user=user,
    )

    pipeline_run_id = get_crewai_pipeline_run_id(user.id)

    async for pipeline_run_info in pipeline_run:
        if not isinstance(pipeline_run_info, PipelineRunStarted) and not isinstance(
            pipeline_run_info, PipelineRunCompleted
        ):
            pipeline_run_info.payload = await get_formatted_graph_data()
            push_to_queue(pipeline_run_id, pipeline_run_info)


async def cognify_github_data(github_data: dict):
    """Process GitHub user, file changes, and comments data from a loaded dictionary."""
    all_datapoints = build_github_datapoints_from_dict(github_data)
    if not all_datapoints:
        logger.error("Failed to create datapoints")
        return False

    dataset_id = uuid5(NAMESPACE_OID, "GitHub")

    cognee_user = await get_default_user()
    tasks = [Task(add_data_points, task_config={"batch_size": 50})]

    await run_with_info_stream(
        tasks=tasks,
        data=all_datapoints,
        dataset_id=dataset_id,
        pipeline_name="github_pipeline",
        user=cognee_user,
    )

    logger.info(f"Done processing {len(all_datapoints)} datapoints")


async def cognify_github_data_from_username(
    username: str,
    token: Optional[str] = None,
    days: int = 30,
    prs_limit: int = 3,
    commits_per_pr: int = 3,
    issues_limit: int = 3,
    max_comments: int = 3,
    skip_no_diff: bool = True,
):
    """Fetches GitHub data for a username and processes it through the DataPoint pipeline."""

    logger.info(f"Fetching GitHub data for user: {username}")

    github_data = get_github_data_for_cognee(
        username=username,
        token=token,
        days=days,
        prs_limit=prs_limit,
        commits_per_pr=commits_per_pr,
        issues_limit=issues_limit,
        max_comments=max_comments,
        skip_no_diff=skip_no_diff,
    )

    if not github_data:
        logger.error(f"Failed to fetch GitHub data for user: {username}")
        return False

    github_data = json.loads(json.dumps(github_data, default=str))

    await cognify_github_data(github_data)


async def process_github_from_file(json_file_path: str):
    """Process GitHub data from a JSON file."""
    logger.info(f"Processing GitHub data from file: {json_file_path}")
    try:
        with open(json_file_path, "r") as f:
            github_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return False

    return await cognify_github_data(github_data)


if __name__ == "__main__":
    import os
    import dotenv

    dotenv.load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    username = ""

    async def cognify_from_username(username, token):
        from cognee.infrastructure.databases.relational import create_db_and_tables

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await create_db_and_tables()
        await cognify_github_data_from_username(username, token)

    # Run it
    asyncio.run(cognify_from_username(username, token))
