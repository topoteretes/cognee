from github import Github
from datetime import datetime
import json
import os
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_dev_comments import GitHubDevComments
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_dev_commits import GitHubDevCommits


class GitHubDevProfile:
    """Class for working with a GitHub developer's profile, commits, and activity."""

    def __init__(self, username, token):
        """Initialize with a username and GitHub API token."""
        self.github = Github(token) if token else Github()
        self.token = token
        self.username = username
        self.user = self._get_user(username)
        self.user_info = self._extract_user_info() if self.user else None
        self.comments = GitHubDevComments(self) if self.user else None
        self.commits = GitHubDevCommits(self) if self.user else None

    def get_user_info(self):
        """Returns the cached user information."""
        return self.user_info

    def get_user_repos(self, limit=None):
        """Returns a list of user's repositories with limit."""
        if not self.user:
            return []

        repos = list(self.user.get_repos())
        if limit:
            repos = repos[:limit]
        return repos

    def get_user_commits(self, days=30, prs_limit=5, commits_per_pr=3, include_files=False):
        """Fetches user's most recent commits from pull requests."""
        if not self.commits:
            return None

        self.commits.set_options(
            days=days,
            prs_limit=prs_limit,
            commits_per_pr=commits_per_pr,
            include_files=include_files,
        )

        return self.commits.get_user_commits()

    def get_user_file_changes(self, days=30, prs_limit=5, commits_per_pr=3, skip_no_diff=True):
        """Returns a flat list of file changes from PRs with associated commit information."""
        if not self.commits:
            return None

        self.commits.set_options(
            days=days,
            prs_limit=prs_limit,
            commits_per_pr=commits_per_pr,
            include_files=True,
            skip_no_diff=skip_no_diff,
        )

        return self.commits.get_user_file_changes()

    def get_issue_comments(
        self, days=30, issues_limit=10, max_comments=5, include_issue_details=True
    ):
        """Fetches comments made by the user on issues across repositories within specified timeframe."""
        if not self.comments:
            return None

        self.comments.set_limits(
            days=days,
            issues_limit=issues_limit,
            max_comments=max_comments,
            include_issue_details=include_issue_details,
        )

        return self.comments.get_issue_comments()

    def get_repo_issue_comments(
        self, repo_name, days=30, issues_limit=10, max_comments=5, include_issue_details=True
    ):
        """Fetches comments made by the user on issues in a specific repository within timeframe."""
        if not self.user or not self.comments:
            return None

        self.comments.set_limits(
            days=days,
            issues_limit=issues_limit,
            max_comments=max_comments,
            include_issue_details=include_issue_details,
        )

        return self.comments.get_repo_issue_comments(repo_name)

    def _get_user(self, username):
        """Fetches a GitHub user object."""
        try:
            return self.github.get_user(username)
        except Exception as e:
            print(f"Error connecting to GitHub API: {e}")
            return None

    def _extract_user_info(self):
        """Extracts basic information from a GitHub user object."""
        return {
            "login": self.user.login,
            "name": self.user.name,
            "bio": self.user.bio,
            "company": self.user.company,
            "location": self.user.location,
            "public_repos": self.user.public_repos,
            "followers": self.user.followers,
            "following": self.user.following,
        }
