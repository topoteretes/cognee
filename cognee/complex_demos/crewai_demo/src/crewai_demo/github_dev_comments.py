from github import Github
from datetime import datetime, timedelta


class GitHubDevComments:
    """Class for working with a GitHub developer's comments."""

    def __init__(
        self, profile, days=30, issues_limit=10, max_comments=5, include_issue_details=True
    ):
        """Initialize with a GitHubDevProfile instance and default parameters."""
        self.profile = profile
        self.days = days
        self.issues_limit = issues_limit
        self.max_comments = max_comments
        self.include_issue_details = include_issue_details

    def get_issue_comments(self):
        """Fetches comments made by the user on issues across repositories within timeframe."""
        if not self.profile.user:
            return None

        date_filter = self._get_date_filter(self.days)
        query = f"commenter:{self.profile.username} is:issue{date_filter}"

        return self._get_comments_from_search(query)

    def get_repo_issue_comments(self, repo_name):
        """Fetches comments made by the user on issues in a specific repository within timeframe."""
        if not self.profile.user:
            return None

        date_filter = self._get_date_filter(self.days)
        query = f"repo:{repo_name} is:issue commenter:{self.profile.username}{date_filter}"
        self.profile.github.get_repo(repo_name)

        return self._get_comments_from_search(query)

    def set_limits(
        self, days=None, issues_limit=None, max_comments=None, include_issue_details=None
    ):
        """Sets all search parameters for comment searches."""
        if days is not None:
            self.days = days
        if issues_limit is not None:
            self.issues_limit = issues_limit
        if max_comments is not None:
            self.max_comments = max_comments
        if include_issue_details is not None:
            self.include_issue_details = include_issue_details

    def _get_date_filter(self, days):
        """Creates a date filter string for GitHub search queries."""
        if not days:
            return ""

        date_limit = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return f" created:>={date_limit}"

    def _get_comments_from_search(self, query):
        """Retrieves comments based on a search query for issues."""
        try:
            issues = list(self.profile.github.search_issues(query))
        except Exception as e:
            print(f"Error executing search query: {e}")
            return []

        if not issues:
            return []

        all_comments = [
            self._extract_comment_data(issue, comment)
            for issue in issues[: self.issues_limit]
            for comment in self._get_user_comments_from_issue(issue)
        ]

        return all_comments

    def _get_user_comments_from_issue(self, issue):
        """Gets comments made by the user on a specific issue."""
        try:
            all_comments = list(issue.get_comments())
            user_comments = [c for c in all_comments if c.user.login == self.profile.username]
            return user_comments[: self.max_comments]
        except Exception as e:
            print(f"Error getting comments from issue #{issue.number}: {e}")
            return []

    def _extract_comment_data(self, issue, comment):
        """Creates a structured data object from a comment."""
        comment_data = {
            "repo": issue.repository.name,
            "issue_number": issue.number,
            "comment_id": comment.id,
            "body": comment.body,
            "created_at": comment.created_at,
            "updated_at": comment.updated_at,
            "html_url": comment.html_url,
            "issue_url": issue.html_url,
            "author_association": getattr(comment, "author_association", "UNKNOWN"),
            "issue_title": issue.title,
            "issue_state": issue.state,
        }

        return comment_data
