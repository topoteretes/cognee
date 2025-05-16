from github import Github
from datetime import datetime, timedelta
import requests
from abc import ABC, abstractmethod


GITHUB_API_URL = "https://api.github.com/graphql"


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
            print(f"Error fetching {self._get_comment_type()} comments: {e}")
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


class IssueCommentsProvider(GitHubCommentBase):
    """Provider for GitHub issue comments."""

    QUERY_TEMPLATE = """
    {{
      user(login: "{username}") {{
        issueComments(first: {limit}, orderBy: {{field: UPDATED_AT, direction: DESC}}) {{
          nodes {{
            body
            createdAt
            updatedAt
            url
            issue {{
              number
              title
              url
              repository {{
                nameWithOwner
              }}
              state
            }}
          }}
        }}
      }}
    }}
    """

    def _build_query(self) -> str:
        """Builds the GraphQL query for issue comments."""
        return self.QUERY_TEMPLATE.format(
            username=self.username,
            limit=self.limit * 2,  # Fetch extra to allow for filtering
        )

    def _extract_comments(self, data) -> list:
        """Extracts issue comments from the GraphQL response."""
        return data["user"]["issueComments"]["nodes"]

    def _format_comment(self, comment) -> dict:
        """Formats an issue comment from GraphQL."""
        # Extract comment ID from URL
        comment_id = comment["url"].split("/")[-1] if comment["url"] else None

        return {
            "repo": comment["issue"]["repository"]["nameWithOwner"],
            "issue_number": comment["issue"]["number"],
            "comment_id": comment_id,
            "body": comment["body"],
            "created_at": comment["createdAt"],
            "updated_at": comment["updatedAt"],
            "html_url": comment["url"],
            "issue_url": comment["issue"]["url"],
            "author_association": "COMMENTER",
            "issue_title": comment["issue"]["title"],
            "issue_state": comment["issue"]["state"],
            "login": self.username,
            "type": "issue_comment",
        }

    def _get_comment_type(self) -> str:
        """Returns the comment type for error messages."""
        return "issue"


class PrReviewsProvider(GitHubCommentBase):
    """Provider for GitHub PR reviews."""

    QUERY_TEMPLATE = """
    {{
      user(login: "{username}") {{
        contributionsCollection(from: "{from_date}", to: "{to_date}") {{
          pullRequestReviewContributions(first: 100) {{
            nodes {{
              pullRequestReview {{
                body
                createdAt
                updatedAt
                url
                state
                pullRequest {{
                  number
                  title
                  url
                  repository {{
                    nameWithOwner
                  }}
                  state
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

    def _build_query(self) -> str:
        """Builds the GraphQL query for PR reviews."""
        today = datetime.now()
        from_date = (today - timedelta(days=365)).strftime("%Y-%m-%dT00:00:00Z")
        to_date = today.strftime("%Y-%m-%dT23:59:59Z")

        return self.QUERY_TEMPLATE.format(
            username=self.username, from_date=from_date, to_date=to_date
        )

    def _extract_comments(self, data) -> list:
        """Extracts PR reviews from the GraphQL response."""
        contributions = data["user"]["contributionsCollection"]["pullRequestReviewContributions"][
            "nodes"
        ]
        return [
            node["pullRequestReview"] for node in contributions if node["pullRequestReview"]["body"]
        ]

    def _format_comment(self, review) -> dict:
        """Formats a PR review from GraphQL."""
        # Extract review ID from URL
        review_id = review["url"].split("/")[-1] if review["url"] else None

        return {
            "repo": review["pullRequest"]["repository"]["nameWithOwner"],
            "issue_number": review["pullRequest"]["number"],
            "comment_id": review_id,
            "body": review["body"],
            "created_at": review["createdAt"],
            "updated_at": review["updatedAt"],
            "html_url": review["url"],
            "issue_url": review["pullRequest"]["url"],
            "author_association": "COMMENTER",
            "issue_title": review["pullRequest"]["title"],
            "issue_state": review["pullRequest"]["state"],
            "login": self.username,
            "review_state": review["state"],
            "type": "pr_review",
        }

    def _get_comment_type(self) -> str:
        """Returns the comment type for error messages."""
        return "PR review"


class PrReviewCommentsProvider(GitHubCommentBase):
    """Provider for GitHub PR review comments (inline code comments)."""

    # Query to get PRs the user has reviewed
    PR_CONTRIBUTIONS_TEMPLATE = """
    {{
      user(login: "{username}") {{
        contributionsCollection(from: "{from_date}", to: "{to_date}") {{
          pullRequestReviewContributions(first: 100) {{
            nodes {{
              pullRequestReview {{
                pullRequest {{
                  number
                  title
                  url
                  repository {{
                    nameWithOwner
                  }}
                  state
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

    # Query to get comments for a specific PR in a specific repository
    PR_COMMENTS_TEMPLATE = """
    {{
      repository(owner: "{owner}", name: "{repo}") {{
        pullRequest(number: {pr_number}) {{
          reviews(first: 100, author: "{username}") {{
            nodes {{
              comments(first: 50) {{
                nodes {{
                  body
                  createdAt
                  updatedAt
                  url
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

    def _build_query(self) -> str:
        """Builds the GraphQL query for PR contributions."""
        today = datetime.now()
        from_date = (today - timedelta(days=365)).strftime("%Y-%m-%dT00:00:00Z")
        to_date = today.strftime("%Y-%m-%dT23:59:59Z")

        return self.PR_CONTRIBUTIONS_TEMPLATE.format(
            username=self.username, from_date=from_date, to_date=to_date
        )

    def _extract_comments(self, data) -> list:
        """Extracts PR review comments using a two-step approach."""
        prs = self._get_reviewed_prs(data)
        return self._fetch_comments_for_prs(prs)

    def _get_reviewed_prs(self, data) -> list:
        """Gets a deduplicated list of PRs the user has reviewed."""
        contributions = data["user"]["contributionsCollection"]["pullRequestReviewContributions"][
            "nodes"
        ]
        unique_prs = []

        for node in contributions:
            pr = node["pullRequestReview"]["pullRequest"]
            if not any(existing_pr["url"] == pr["url"] for existing_pr in unique_prs):
                unique_prs.append(pr)

        return unique_prs[: min(10, len(unique_prs))]

    def _fetch_comments_for_prs(self, prs) -> list:
        """Fetches inline comments for each PR in the list."""
        all_comments = []

        for pr in prs:
            comments = self._get_comments_for_pr(pr)
            all_comments.extend(comments)

        return all_comments

    def _get_comments_for_pr(self, pr) -> list:
        """Fetches the inline comments for a specific PR."""
        owner, repo = pr["repository"]["nameWithOwner"].split("/")

        pr_query = self.PR_COMMENTS_TEMPLATE.format(
            owner=owner, repo=repo, pr_number=pr["number"], username=self.username
        )

        try:
            pr_comments = []
            pr_data = self._run_query(pr_query)
            reviews = pr_data["repository"]["pullRequest"]["reviews"]["nodes"]

            for review in reviews:
                for comment in review["comments"]["nodes"]:
                    comment["_pr_data"] = pr
                    pr_comments.append(comment)

            return pr_comments
        except Exception as e:
            print(f"Error fetching comments for PR #{pr['number']}: {e}")
            return []

    def _format_comment(self, comment) -> dict:
        """Formats a PR review comment from GraphQL."""
        pr = comment["_pr_data"]
        # Extract comment ID from URL
        comment_id = comment["url"].split("/")[-1] if comment["url"] else None

        return {
            "repo": pr["repository"]["nameWithOwner"],
            "issue_number": pr["number"],
            "comment_id": comment_id,
            "body": comment["body"],
            "created_at": comment["createdAt"],
            "updated_at": comment["updatedAt"],
            "html_url": comment["url"],
            "issue_url": pr["url"],
            "author_association": "COMMENTER",
            "issue_title": pr["title"],
            "issue_state": pr["state"],
            "login": self.username,
            "type": "pr_review_comment",
        }

    def _get_comment_type(self) -> str:
        """Returns the comment type for error messages."""
        return "PR review comment"


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

        # Initialize providers
        issue_provider = IssueCommentsProvider(
            self.profile.token, self.profile.username, self.limit
        )
        pr_review_provider = PrReviewsProvider(
            self.profile.token, self.profile.username, self.limit
        )
        pr_comment_provider = PrReviewCommentsProvider(
            self.profile.token, self.profile.username, self.limit
        )

        # Get comments from all providers
        issue_comments = issue_provider.get_comments()
        pr_reviews = pr_review_provider.get_comments()
        pr_review_comments = pr_comment_provider.get_comments()

        # Combine all comments
        return issue_comments + pr_reviews + pr_review_comments

    def set_limit(self, limit=None, include_issue_details=None):
        """Sets the limit for comments to retrieve."""
        if limit is not None:
            self.limit = limit
        if include_issue_details is not None:
            self.include_issue_details = include_issue_details


if __name__ == "__main__":
    import os
    from cognee.complex_demos.crewai_demo.src.crewai_demo.github_dev_profile import GitHubDevProfile

    # Get GitHub token from environment variable
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Please set the GITHUB_TOKEN environment variable")
        exit(1)

    # Replace with the GitHub username you want to test
    username = "hajdul88"  # Replace with actual username to test

    # Initialize profile and fetch comments
    profile = GitHubDevProfile(username, token)

    # Get comments from both issues and PRs using GraphQL
    comments = profile.get_issue_comments(limit=5)

    # Group by type
    issue_comments = [c for c in comments if c.get("type") == "issue_comment"]
    pr_reviews = [c for c in comments if c.get("type") == "pr_review"]
    pr_review_comments = [c for c in comments if c.get("type") == "pr_review_comment"]

    # Print results
    print(f"Found {len(comments)} comments by {username}:")
    print(f"- {len(issue_comments)} issue comments")
    print(f"- {len(pr_reviews)} PR reviews")
    print(f"- {len(pr_review_comments)} PR review comments")

    for i, comment in enumerate(comments, 1):
        comment_type = (
            "Issue Comment"
            if comment.get("type") == "issue_comment"
            else "PR Review"
            if comment.get("type") == "pr_review"
            else "PR Review Comment"
        )
        print(f"\n{i}. {comment_type} in {comment['repo']} - {comment['issue_title']}")
        print(f"   URL: {comment['html_url']}")
        print(f"   Created: {comment['created_at']}")

        if comment.get("type") == "pr_review":
            print(f"   Review State: {comment.get('review_state', 'N/A')}")

        body = comment["body"]
        if body:
            print(f"   Body: {body[:100]}..." if len(body) > 100 else f"   Body: {body}")
        else:
            print("   Body: [No content]")
