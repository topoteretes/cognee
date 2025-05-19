from datetime import datetime, timedelta
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_comment_base import (
    GitHubCommentBase,
    logger,
)


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
        return self.QUERY_TEMPLATE.format(username=self.username, limit=self.limit)

    def _extract_comments(self, data) -> list:
        """Extracts issue comments from the GraphQL response."""
        return data["user"]["issueComments"]["nodes"]

    def _format_comment(self, comment) -> dict:
        """Formats an issue comment from GraphQL."""
        comment_id = comment["url"].split("/")[-1] if comment["url"] else None

        return {
            "repo": comment["issue"]["repository"]["nameWithOwner"],
            "issue_number": comment["issue"]["number"],
            "comment_id": comment_id,
            "body": comment["body"],
            "text": comment["body"],
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
        contributionsCollection {{
          pullRequestReviewContributions(first: {fetch_limit}) {{
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

    def __init__(self, token, username, limit=10, fetch_limit=None):
        """Initialize with token, username, and optional limits."""
        super().__init__(token, username, limit)
        self.fetch_limit = fetch_limit if fetch_limit is not None else 10 * limit

    def _build_query(self) -> str:
        """Builds the GraphQL query for PR reviews."""
        return self.QUERY_TEMPLATE.format(username=self.username, fetch_limit=self.fetch_limit)

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
        review_id = review["url"].split("/")[-1] if review["url"] else None

        return {
            "repo": review["pullRequest"]["repository"]["nameWithOwner"],
            "issue_number": review["pullRequest"]["number"],
            "comment_id": review_id,
            "body": review["body"],
            "text": review["body"],
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

    PR_CONTRIBUTIONS_TEMPLATE = """
    {{
      user(login: "{username}") {{
        contributionsCollection {{
          pullRequestReviewContributions(first: {fetch_limit}) {{
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

    PR_COMMENTS_TEMPLATE = """
    {{
      repository(owner: "{owner}", name: "{repo}") {{
        pullRequest(number: {pr_number}) {{
          reviews(first: {reviews_limit}, author: "{username}") {{
            nodes {{
              comments(first: {comments_limit}) {{
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

    def __init__(
        self,
        token,
        username,
        limit=10,
        fetch_limit=None,
        reviews_limit=None,
        comments_limit=None,
        pr_limit=None,
    ):
        """Initialize with token, username, and optional limits."""
        super().__init__(token, username, limit)
        self.fetch_limit = fetch_limit if fetch_limit is not None else 4 * limit
        self.reviews_limit = reviews_limit if reviews_limit is not None else 2 * limit
        self.comments_limit = comments_limit if comments_limit is not None else 3 * limit
        self.pr_limit = pr_limit if pr_limit is not None else 2 * limit

    def _build_query(self) -> str:
        """Builds the GraphQL query for PR contributions."""
        return self.PR_CONTRIBUTIONS_TEMPLATE.format(
            username=self.username, fetch_limit=self.fetch_limit
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

        return unique_prs[: min(self.pr_limit, len(unique_prs))]

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
            owner=owner,
            repo=repo,
            pr_number=pr["number"],
            username=self.username,
            reviews_limit=self.reviews_limit,
            comments_limit=self.comments_limit,
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
            logger.error(f"Error fetching comments for PR #{pr['number']}: {e}")
            return []

    def _format_comment(self, comment) -> dict:
        """Formats a PR review comment from GraphQL."""
        pr = comment["_pr_data"]
        comment_id = comment["url"].split("/")[-1] if comment["url"] else None

        return {
            "repo": pr["repository"]["nameWithOwner"],
            "issue_number": pr["number"],
            "comment_id": comment_id,
            "body": comment["body"],
            "text": comment["body"],
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
