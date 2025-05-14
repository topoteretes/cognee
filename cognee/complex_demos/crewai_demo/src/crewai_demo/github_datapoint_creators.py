from uuid import uuid5, NAMESPACE_OID
from typing import Dict, Any, List, Tuple, Optional

from cognee.low_level import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from cognee.shared.logging_utils import get_logger
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_datapoints import (
    GitHubUser,
    Repository,
    File,
    FileChange,
    Comment,
    Issue,
    Commit,
)

logger = get_logger("github_datapoints")


def create_github_user_datapoint(user_data, nodesets: List[NodeSet]):
    """Creates just the GitHubUser DataPoint object from the user data, with node sets."""
    if not user_data:
        return None

    user_id = uuid5(NAMESPACE_OID, user_data.get("login", ""))

    user = GitHubUser(
        id=user_id,
        login=user_data.get("login", ""),
        name=user_data.get("name"),
        bio=user_data.get("bio"),
        company=user_data.get("company"),
        location=user_data.get("location"),
        public_repos=user_data.get("public_repos", 0),
        followers=user_data.get("followers", 0),
        following=user_data.get("following", 0),
        interacts_with=[],
        belongs_to_set=nodesets,
    )

    logger.debug(f"Created GitHubUser with ID: {user_id}")

    return [user] + nodesets


def create_repository_datapoint(repo_name: str, nodesets: List[NodeSet]) -> Repository:
    """Creates a Repository DataPoint with a consistent ID."""
    repo_id = uuid5(NAMESPACE_OID, repo_name)
    repo = Repository(
        id=repo_id,
        name=repo_name,
        has_issue=[],
        has_commit=[],
        contains=[],
        belongs_to_set=nodesets,
    )
    logger.debug(f"Created Repository with ID: {repo_id} for {repo_name}")
    return repo


def create_file_datapoint(filename: str, repo_name: str, nodesets: List[NodeSet]) -> File:
    """Creates a File DataPoint with a consistent ID."""
    file_key = f"{repo_name}:{filename}"
    file_id = uuid5(NAMESPACE_OID, file_key)
    file = File(id=file_id, filename=filename, repo=repo_name, belongs_to_set=nodesets)
    logger.debug(f"Created File with ID: {file_id} for {filename}")
    return file


def create_commit_datapoint(
    commit_data: Dict[str, Any], user: GitHubUser, nodesets: List[NodeSet]
) -> Commit:
    """Creates a Commit DataPoint with a consistent ID and connection to user."""
    commit_id = uuid5(NAMESPACE_OID, commit_data.get("commit_sha", ""))
    commit = Commit(
        id=commit_id,
        commit_sha=commit_data.get("commit_sha", ""),
        commit_message=commit_data.get("commit_message", ""),
        commit_date=commit_data.get("commit_date", ""),
        commit_url=commit_data.get("commit_url", ""),
        author_name=commit_data.get("login", ""),
        repo=commit_data.get("repo", ""),
        authored_by=user,
        has_change=[],
        belongs_to_set=nodesets,
    )
    logger.debug(f"Created Commit with ID: {commit_id} for {commit_data.get('commit_sha', '')}")
    return commit


def create_file_change_datapoint(
    fc_data: Dict[str, Any], file: File, nodesets: List[NodeSet]
) -> FileChange:
    """Creates a FileChange DataPoint with a consistent ID."""
    fc_key = (
        f"{fc_data.get('repo', '')}:{fc_data.get('commit_sha', '')}:{fc_data.get('filename', '')}"
    )
    fc_id = uuid5(NAMESPACE_OID, fc_key)

    file_change = FileChange(
        id=fc_id,
        filename=fc_data.get("filename", ""),
        status=fc_data.get("status", ""),
        additions=fc_data.get("additions", 0),
        deletions=fc_data.get("deletions", 0),
        changes=fc_data.get("changes", 0),
        diff=fc_data.get("diff", ""),
        commit_sha=fc_data.get("commit_sha", ""),
        repo=fc_data.get("repo", ""),
        modifies=file,
        belongs_to_set=nodesets,
    )
    logger.debug(f"Created FileChange with ID: {fc_id} for {fc_data.get('filename', '')}")
    return file_change


def create_issue_datapoint(
    issue_data: Dict[str, Any], repo_name: str, nodesets: List[NodeSet]
) -> Issue:
    """Creates an Issue DataPoint with a consistent ID."""
    issue_key = f"{repo_name}:{issue_data.get('issue_number', '')}"
    issue_id = uuid5(NAMESPACE_OID, issue_key)

    issue = Issue(
        id=issue_id,
        number=issue_data.get("issue_number", 0),
        title=issue_data.get("issue_title", ""),
        state=issue_data.get("issue_state", ""),
        repository=repo_name,
        is_pr=False,
        has_comment=[],
        belongs_to_set=nodesets,
    )
    logger.debug(f"Created Issue with ID: {issue_id} for {issue_data.get('issue_title', '')}")
    return issue


def create_comment_datapoint(
    comment_data: Dict[str, Any], user: GitHubUser, nodesets: List[NodeSet]
) -> Comment:
    """Creates a Comment DataPoint with a consistent ID and connection to user."""
    comment_key = f"{comment_data.get('repo', '')}:{comment_data.get('issue_number', '')}:{comment_data.get('comment_id', '')}"
    comment_id = uuid5(NAMESPACE_OID, comment_key)

    comment = Comment(
        id=comment_id,
        comment_id=str(comment_data.get("comment_id", "")),
        body=comment_data.get("body", ""),
        created_at=comment_data.get("created_at", ""),
        updated_at=comment_data.get("updated_at", ""),
        author_name=comment_data.get("login", ""),
        issue_number=comment_data.get("issue_number", 0),
        repo=comment_data.get("repo", ""),
        authored_by=user,
        belongs_to_set=nodesets,
    )
    logger.debug(f"Created Comment with ID: {comment_id}")
    return comment


def create_github_datapoints(github_data, nodesets: List[NodeSet]):
    """Creates DataPoint objects from GitHub data - simplified to just create user for now."""
    if not github_data:
        return None

    return create_github_user_datapoint(github_data["user"], nodesets)
