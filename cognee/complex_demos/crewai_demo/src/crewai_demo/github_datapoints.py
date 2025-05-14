from uuid import uuid5, NAMESPACE_OID
from typing import Optional, List
from cognee.low_level import DataPoint
from cognee.modules.engine.models.node_set import NodeSet


class File(DataPoint):
    """File is now a leaf node without any lists of other DataPoints"""

    filename: str
    repo: str
    metadata: dict = {"index_fields": ["filename"]}


class GitHubUser(DataPoint):
    login: str
    name: Optional[str]
    bio: Optional[str]
    company: Optional[str]
    location: Optional[str]
    public_repos: int
    followers: int
    following: int
    interacts_with: List["Repository"] = []
    metadata: dict = {"index_fields": ["login"]}


class FileChange(DataPoint):
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    diff: str
    commit_sha: str
    repo: str
    modifies: File
    metadata: dict = {"index_fields": ["diff"]}


class Comment(DataPoint):
    comment_id: str
    body: str
    created_at: str
    updated_at: str
    author_name: str
    issue_number: int
    repo: str
    authored_by: GitHubUser
    metadata: dict = {"index_fields": ["body"]}


class Issue(DataPoint):
    number: int
    title: str
    state: str
    repository: str
    is_pr: bool
    has_comment: List[Comment] = []
    metadata: dict = {"index_fields": ["title"]}


class Commit(DataPoint):
    commit_sha: str
    commit_message: str
    commit_date: str
    commit_url: str
    author_name: str
    repo: str
    authored_by: GitHubUser
    has_change: List[FileChange] = []
    metadata: dict = {"index_fields": ["commit_message"]}


class Repository(DataPoint):
    name: str
    has_issue: List[Issue] = []
    has_commit: List[Commit] = []
    contains: List[File] = []
    metadata: dict = {"index_fields": ["name"]}
