from github import Github
from datetime import datetime, timedelta


class GitHubDevCommits:
    """Class for working with a GitHub developer's commits in pull requests."""

    def __init__(
        self,
        profile,
        days=30,
        prs_limit=10,
        commits_per_pr=5,
        include_files=False,
        skip_no_diff=False,
    ):
        """Initialize with a GitHubDevProfile instance and default parameters."""
        self.profile = profile
        self.days = days
        self.prs_limit = prs_limit
        self.commits_per_pr = commits_per_pr
        self.include_files = include_files
        self.skip_no_diff = skip_no_diff
        self.file_keys = ["filename", "status", "additions", "deletions", "changes", "diff"]

    def get_user_commits(self):
        """Fetches user's most recent commits from pull requests."""
        if not self.profile.user:
            return None

        commits = self._collect_user_pr_commits()
        return {"user": self.profile.get_user_info(), "commits": commits}

    def get_user_file_changes(self):
        """Returns a flat list of file changes with associated commit information from PRs."""
        if not self.profile.user:
            return None

        all_files = []
        commits = self._collect_user_pr_commits(include_files=True)

        for commit in commits:
            if "files" not in commit:
                continue

            commit_info = {
                "repo": commit["repo"],
                "commit_sha": commit["sha"],
                "commit_message": commit["message"],
                "commit_date": commit["date"],
                "commit_url": commit["url"],
                "pr_number": commit.get("pr_number"),
                "pr_title": commit.get("pr_title"),
            }

            file_changes = []
            for file in commit["files"]:
                file_data = {key: file.get(key) for key in self.file_keys}
                file_changes.append({**file_data, **commit_info})

            all_files.extend(file_changes)

        return all_files

    def set_options(
        self, days=None, prs_limit=None, commits_per_pr=None, include_files=None, skip_no_diff=None
    ):
        """Sets commit search parameters."""
        if days is not None:
            self.days = days
        if prs_limit is not None:
            self.prs_limit = prs_limit
        if commits_per_pr is not None:
            self.commits_per_pr = commits_per_pr
        if include_files is not None:
            self.include_files = include_files
        if skip_no_diff is not None:
            self.skip_no_diff = skip_no_diff

    def _get_date_filter(self, days):
        """Creates a date filter string for GitHub search queries."""
        if not days:
            return ""

        date_limit = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return f" created:>={date_limit}"

    def _collect_user_pr_commits(self, include_files=None):
        """Collects and sorts a user's recent commits from pull requests they authored."""
        include_files = include_files if include_files is not None else self.include_files

        prs = self._get_user_prs()

        all_commits = []
        for pr in prs[: self.prs_limit]:
            pr_commits = self._get_commits_from_pr(pr, include_files)
            all_commits.extend(pr_commits)

        sorted_commits = sorted(all_commits, key=lambda x: x["date"], reverse=True)
        return sorted_commits

    def _get_user_prs(self):
        """Gets pull requests authored by the user."""
        date_filter = self._get_date_filter(self.days)
        query = f"author:{self.profile.username} is:pr is:merged{date_filter}"

        try:
            return list(self.profile.github.search_issues(query))
        except Exception as e:
            print(f"Error searching for PRs: {e}")
            return []

    def _get_commits_from_pr(self, pr_issue, include_files=None):
        """Gets commits by the user from a specific PR."""
        include_files = include_files if include_files is not None else self.include_files

        pr_info = self._get_pull_request_object(pr_issue)
        if not pr_info:
            return []

        repo_name, pr = pr_info

        all_commits = self._get_all_pr_commits(pr, pr_issue.number)
        if not all_commits:
            return []

        user_commits = [
            c
            for c in all_commits
            if c.author and hasattr(c.author, "login") and c.author.login == self.profile.username
        ]

        commit_data = [
            self._extract_commit_data(commit, repo_name, pr_issue, include_files)
            for commit in user_commits[: self.commits_per_pr]
        ]

        return commit_data

    def _get_pull_request_object(self, pr_issue):
        """Gets repository and pull request objects from an issue."""
        try:
            repo_name = pr_issue.repository.full_name
            repo = self.profile.github.get_repo(repo_name)
            pr = repo.get_pull(pr_issue.number)
            return (repo_name, pr)
        except Exception as e:
            print(f"Error accessing PR #{pr_issue.number}: {e}")
            return None

    def _get_all_pr_commits(self, pr, pr_number):
        """Gets all commits from a pull request."""
        try:
            return list(pr.get_commits())
        except Exception as e:
            print(f"Error retrieving commits from PR #{pr_number}: {e}")
            return None

    def _extract_commit_data(self, commit, repo_name, pr_issue, include_files=None):
        """Extracts relevant data from a commit object within a PR context."""
        commit_data = {
            "repo": repo_name,
            "sha": commit.sha,
            "message": commit.commit.message,
            "date": commit.commit.author.date,
            "url": commit.html_url,
            "pr_number": pr_issue.number,
            "pr_title": pr_issue.title,
            "pr_url": pr_issue.html_url,
        }

        include_files = include_files if include_files is not None else self.include_files

        if include_files:
            commit_data["files"] = self._extract_commit_files(commit)

        return commit_data

    def _extract_commit_files(self, commit):
        """Extracts files changed in a commit, including diffs."""
        files = []
        for file in commit.files:
            if self.skip_no_diff and not file.patch:
                continue

            file_data = {key: getattr(file, key, None) for key in self.file_keys}

            if "diff" in self.file_keys:
                file_data["diff"] = file.patch if file.patch else "No diff available for this file"

            files.append(file_data)
        return files
