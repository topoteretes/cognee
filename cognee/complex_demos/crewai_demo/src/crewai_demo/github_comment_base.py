from abc import ABC, abstractmethod
import requests
from cognee.shared.logging_utils import get_logger

GITHUB_API_URL = "https://api.github.com/graphql"

logger = get_logger("github_comments")


class GitHubCommentBase(ABC):
    """Base class for GitHub comment providers."""

    def __init__(self, token, username, limit=10):
        self.token = token
        self.username = username
        self.limit = limit

    def _run_query(self, query: str) -> dict:
        """Executes a GraphQL query against GitHub's API."""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(GITHUB_API_URL, json={"query": query}, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.status_code} - {response.text}")
        return response.json()["data"]

    def get_comments(self):
        """Template method that orchestrates the comment retrieval process."""
        try:
            query = self._build_query()
            data = self._run_query(query)
            raw_comments = self._extract_comments(data)
            return [self._format_comment(item) for item in raw_comments[: self.limit]]
        except Exception as e:
            logger.error(f"Error fetching {self._get_comment_type()} comments: {e}")
            return []

    @abstractmethod
    def _build_query(self) -> str:
        """Builds the GraphQL query string."""
        pass

    @abstractmethod
    def _extract_comments(self, data) -> list:
        """Extracts the comment data from the GraphQL response."""
        pass

    @abstractmethod
    def _format_comment(self, item) -> dict:
        """Formats a single comment."""
        pass

    @abstractmethod
    def _get_comment_type(self) -> str:
        """Returns the type of comment this provider handles."""
        pass
