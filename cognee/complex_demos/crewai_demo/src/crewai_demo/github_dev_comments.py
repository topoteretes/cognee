from github import Github
from datetime import datetime
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_comment_providers import (
    IssueCommentsProvider,
    PrReviewsProvider,
    PrReviewCommentsProvider,
)


class GitHubDevComments:
    """Facade class for working with a GitHub developer's comments."""

    def __init__(self, profile, limit=10, include_issue_details=True):
        """Initialize with a GitHubDevProfile instance and default parameters."""
        self.profile = profile
        self.limit = limit
        self.include_issue_details = include_issue_details

    def get_issue_comments(self):
        """Fetches the most recent comments made by the user on issues and PRs across repositories."""
        if not self.profile.user:
            return None

        # Calculate all limits based on the base limit
        fetch_limit = self.limit * 4
        reviews_limit = self.limit * 2
        comments_limit = self.limit * 3
        pr_limit = self.limit * 2

        issue_provider = IssueCommentsProvider(
            self.profile.token, self.profile.username, self.limit
        )
        pr_review_provider = PrReviewsProvider(
            self.profile.token, self.profile.username, self.limit, fetch_limit=fetch_limit
        )
        pr_comment_provider = PrReviewCommentsProvider(
            self.profile.token,
            self.profile.username,
            self.limit,
            fetch_limit=fetch_limit,
            reviews_limit=reviews_limit,
            comments_limit=comments_limit,
            pr_limit=pr_limit,
        )

        issue_comments = issue_provider.get_comments()
        pr_reviews = pr_review_provider.get_comments()
        pr_review_comments = pr_comment_provider.get_comments()

        return issue_comments + pr_reviews + pr_review_comments

    def set_limit(self, limit=None, include_issue_details=None):
        """Sets the limit for comments to retrieve."""
        if limit is not None:
            self.limit = limit
        if include_issue_details is not None:
            self.include_issue_details = include_issue_details
