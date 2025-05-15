from uuid import uuid5, NAMESPACE_OID
from typing import Optional, List
from cognee.infrastructure.engine import DataPoint


class File(DataPoint):
    """File is now a leaf node without any lists of other DataPoints"""

    filename: str
    repo: str
    metadata: dict = {"index_fields": ["filename"]}


class GitHubUser(DataPoint):
    name: Optional[str]
    bio: Optional[str]
    company: Optional[str]
    location: Optional[str]
    public_repos: int
    followers: int
    following: int
    interacts_with: List["Repository"] = []
    metadata: dict = {"index_fields": ["name"]}


class FileChange(DataPoint):
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    text: str
    commit_sha: str
    repo: str
    modifies: str
    changed_by: GitHubUser
    metadata: dict = {"index_fields": ["text"]}


class Comment(DataPoint):
    comment_id: str
    text: str
    created_at: str
    updated_at: str
    author_name: str
    issue_number: int
    repo: str
    authored_by: GitHubUser
    metadata: dict = {"index_fields": ["text"]}


class Issue(DataPoint):
    number: int
    text: str
    state: str
    repository: str
    is_pr: bool
    has_comment: List[Comment] = []


class Commit(DataPoint):
    commit_sha: str
    text: str
    commit_date: str
    commit_url: str
    author_name: str
    repo: str
    has_change: List[FileChange] = []


class Repository(DataPoint):
    name: str
    has_issue: List[Issue] = []
    has_commit: List[Commit] = []
