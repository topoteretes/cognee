"""
Fetch and analyze pull requests and discussions from GitHub repositories.
"""
import logging
import asyncio
from typing import AsyncGenerator, Dict, List, Any
import aiohttp
from uuid import uuid4, NAMESPACE_OID, uuid5
from datetime import datetime
import re

from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, RelationshipType, Edge
from cognee.tasks.github.fetch_repositories import Repository, Developer, fetch_github_data

logger = logging.getLogger(__name__)

class PullRequest(DataPoint):
    """Pull request entity from GitHub."""
    
    class Meta:
        category = "github"
        index_fields = ["title", "body"]
    
    def __init__(
        self,
        id,
        number: int,
        title: str,
        body: str = None,
        state: str = None,
        repository_id: str = None,
        repository_name: str = None,
        created_at: str = None,
        updated_at: str = None,
        closed_at: str = None,
        merged_at: str = None,
        creator_id: str = None,
        creator_username: str = None,
        url: str = None,
        html_url: str = None,
        api_url: str = None,
        **kwargs
    ):
        # Create a human-readable display name for the PR
        display_name = f"PR #{number}: {title}"
        
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "number": number,
            "title": title,
            "body": body,
            "state": state,
            "repository_id": repository_id,
            "repository_name": repository_name,
            "created_at": created_at,
            "updated_at": updated_at,
            "closed_at": closed_at,
            "merged_at": merged_at,
            "creator_id": creator_id,
            "creator_username": creator_username,
            "url": url,
            "html_url": html_url,
            "api_url": api_url,
            "name": display_name,  # Add a name field for graph visualization
            **kwargs
        }
        super().__init__(**data)


class PRComment(DataPoint):
    """Pull request comment entity from GitHub."""
    
    class Meta:
        category = "github"
        index_fields = ["body"]
    
    def __init__(
        self,
        id,
        body: str,
        pull_request_id: str = None,
        pull_request_number: int = None,
        repository_id: str = None,
        created_at: str = None,
        updated_at: str = None,
        author_id: str = None,
        author_username: str = None,
        url: str = None,
        html_url: str = None,
        api_url: str = None,
        **kwargs
    ):
        # Create a human-readable display name for the comment
        comment_preview = body[:40] + "..." if len(body) > 40 else body
        display_name = f"Comment by {author_username}: {comment_preview}"
        
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "body": body,
            "pull_request_id": pull_request_id,
            "pull_request_number": pull_request_number,
            "repository_id": repository_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "author_id": author_id,
            "author_username": author_username,
            "url": url,
            "html_url": html_url,
            "api_url": api_url,
            "name": display_name,  # Add a name field for graph visualization
            **kwargs
        }
        super().__init__(**data)


class DeveloperInteraction(DataPoint):
    """Entity representing interactions between developers in PR discussions."""
    
    class Meta:
        category = "github"
        index_fields = ["interaction_type", "sentiment", "developer1_username", "developer2_username"]
    
    def __init__(
        self,
        id,
        interaction_type: str,  # 'review', 'comment', 'reply'
        developer1_id: str,
        developer2_id: str,
        developer1_username: str,
        developer2_username: str,
        sentiment: str = "neutral",  # 'positive', 'negative', 'neutral'
        response_time_seconds: int = None,
        pull_request_id: str = None,
        repository_id: str = None,
        comment_id: str = None,
        date: str = None,
        **kwargs
    ):
        # Create a human-readable display name for the interaction
        display_name = f"{interaction_type.capitalize()}: {developer1_username} → {developer2_username} ({sentiment})"
        
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "interaction_type": interaction_type,
            "developer1_id": developer1_id,
            "developer2_id": developer2_id,
            "developer1_username": developer1_username,
            "developer2_username": developer2_username,
            "sentiment": sentiment,
            "response_time_seconds": response_time_seconds,
            "pull_request_id": pull_request_id,
            "repository_id": repository_id,
            "comment_id": comment_id,
            "date": date,
            "name": display_name,  # Add a name field for graph visualization
            **kwargs
        }
        super().__init__(**data)


async def fetch_pull_requests(repo: Repository, api_token: str = None, max_prs: int = 5, max_comments_per_pr: int = 20) -> AsyncGenerator[DataPoint, None]:
    """
    Fetch pull requests for a given repository and analyze interactions between developers.
    Limited to a certain number of PRs and comments per PR.
    
    Args:
        repo: Repository object
        api_token: GitHub API token for authentication
        max_prs: Maximum number of PRs to fetch (default: 5)
        max_comments_per_pr: Maximum number of comments to fetch per PR (default: 20)
    """
    headers = {}
    if api_token:
        headers["Authorization"] = f"token {api_token}"
    
    # Dictionary to store developers by ID
    developers = {}
    
    # Extract just the repo name from the full repo.name which might be in "owner/name" format
    repo_name_parts = repo.name.split('/')
    repo_name = repo_name_parts[-1] if len(repo_name_parts) > 1 else repo.name
    owner = repo.owner or repo_name_parts[0] if len(repo_name_parts) > 1 else "unknown"
    
    # Fetch pull requests
    prs_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls?state=all&sort=updated&direction=desc&per_page={max_prs}"
    prs_data = await fetch_github_data(prs_url, headers)
    
    if not prs_data:
        logger.warning(f"No pull requests found for repository {repo.name}")
        return
    
    # Process only up to max_prs
    for pr_data in prs_data[:max_prs]:
        pr_id = uuid5(NAMESPACE_OID, str(pr_data.get("id", "")))
        pr_number = pr_data.get("number")
        
        # Create or get the PR creator
        creator_data = pr_data.get("user", {})
        creator_id = str(uuid5(NAMESPACE_OID, str(creator_data.get("id", ""))))
        creator_username = creator_data.get("login")
        
        if creator_id not in developers:
            # Fetch more detailed developer info
            developer_url = f"https://api.github.com/users/{creator_username}"
            developer_data = await fetch_github_data(developer_url, headers)
            
            developer = Developer(
                id=creator_id,
                username=creator_username,
                name=developer_data.get("name"),
                email=developer_data.get("email"),
                avatar_url=developer_data.get("avatar_url"),
                html_url=developer_data.get("html_url"),
                api_url=developer_data.get("url"),
                company=developer_data.get("company"),
                location=developer_data.get("location"),
                bio=developer_data.get("bio"),
                created_at=developer_data.get("created_at")
            )
            
            yield developer
            developers[creator_id] = developer
        
        # Create pull request entity
        pr = PullRequest(
            id=pr_id,
            number=pr_number,
            title=pr_data.get("title"),
            body=pr_data.get("body", ""),
            state=pr_data.get("state"),
            repository_id=str(repo.id),
            repository_name=repo.name,
            created_at=pr_data.get("created_at"),
            updated_at=pr_data.get("updated_at"),
            closed_at=pr_data.get("closed_at"),
            merged_at=pr_data.get("merged_at"),
            creator_id=creator_id,
            creator_username=creator_username,
            url=pr_data.get("url"),
            html_url=pr_data.get("html_url"),
            api_url=pr_data.get("url"),
            additions=pr_data.get("additions"),
            deletions=pr_data.get("deletions"),
            changed_files=pr_data.get("changed_files")
        )
        
        yield pr
        
        # Create authored relationship
        authored_edge = Edge(
            source_node_id=str(developers[creator_id].id),
            target_node_id=str(pr.id),
            relationship_name=RelationshipType.AUTHORED.value,
            properties={
                "weight": 1.0,
                "name": f"{creator_username} authored PR #{pr_number}"  # Add edge name
            }
        )
        
        yield authored_edge
        
        # Create PR to Repo relationship
        pr_repo_edge = Edge(
            source_node_id=str(pr.id),
            target_node_id=str(repo.id),
            relationship_name=RelationshipType.BELONGS_TO.value,
            properties={
                "weight": 1.0,
                "name": f"PR #{pr_number} belongs to {repo.name}"  # Add edge name
            }
        )
        
        yield pr_repo_edge
        
        # Fetch PR comments and reviews
        await asyncio.sleep(0.5)  # Rate limiting precaution
        
        # Fetch PR review comments with limit
        async for datapoint in fetch_pr_comments(repo, pr, developers, headers, max_comments_per_pr):
            yield datapoint


async def fetch_pr_comments(
    repo: Repository, 
    pr: PullRequest, 
    developers: Dict[str, Developer], 
    headers: Dict,
    max_comments: int = 20
) -> AsyncGenerator[DataPoint, None]:
    """
    Fetch comments for a pull request and analyze interactions.
    Limited to a maximum number of comments.
    
    Args:
        repo: Repository object
        pr: PullRequest object
        developers: Dictionary of developers by ID
        headers: HTTP headers for GitHub API
        max_comments: Maximum number of comments to fetch (default: 20)
    """
    # Fetch PR comments
    comments_url = f"https://api.github.com/repos/{repo.owner}/{repo.name}/pulls/{pr.number}/comments?per_page={max_comments}"
    comments_data = await fetch_github_data(comments_url, headers)
    
    # Also fetch issue comments for the PR
    issue_comments_url = f"https://api.github.com/repos/{repo.owner}/{repo.name}/issues/{pr.number}/comments?per_page={max_comments}"
    issue_comments_data = await fetch_github_data(issue_comments_url, headers)
    
    # Combine both types of comments and sort by created_at (newest first)
    all_comments = sorted(
        comments_data + issue_comments_data,
        key=lambda x: x.get("created_at", ""),
        reverse=True
    )
    
    if not all_comments:
        return
    
    # Process only up to max_comments
    for comment_data in all_comments[:max_comments]:
        comment_id = uuid5(NAMESPACE_OID, str(comment_data.get("id", "")))
        
        # Get comment author
        author_data = comment_data.get("user", {})
        author_id = str(uuid5(NAMESPACE_OID, str(author_data.get("id", ""))))
        author_username = author_data.get("login")
        
        # Create or get developer
        if author_id not in developers:
            # Fetch more detailed developer info
            developer_url = f"https://api.github.com/users/{author_username}"
            developer_data = await fetch_github_data(developer_url, headers)
            
            # Create a human-readable display name for the developer
            display_name = developer_data.get("name") or author_username
            
            developer = Developer(
                id=author_id,
                username=author_username,
                name=display_name,  # Ensure we use a human-readable name
                email=developer_data.get("email"),
                avatar_url=developer_data.get("avatar_url"),
                html_url=developer_data.get("html_url"),
                api_url=developer_data.get("url"),
                company=developer_data.get("company"),
                location=developer_data.get("location"),
                bio=developer_data.get("bio"),
                created_at=developer_data.get("created_at")
            )
            
            yield developer
            developers[author_id] = developer
        
        # Create comment entity
        comment = PRComment(
            id=comment_id,
            body=comment_data.get("body", ""),
            pull_request_id=str(pr.id),
            pull_request_number=pr.number,
            repository_id=str(repo.id),
            created_at=comment_data.get("created_at"),
            updated_at=comment_data.get("updated_at"),
            author_id=author_id,
            author_username=author_username,
            url=comment_data.get("url"),
            html_url=comment_data.get("html_url"),
            api_url=comment_data.get("url")
        )
        
        yield comment
        
        # Create authored relationship
        authored_edge = Edge(
            source_node_id=str(developers[author_id].id),
            target_node_id=str(comment.id),
            relationship_name=RelationshipType.AUTHORED.value,
            properties={
                "weight": 1.0,
                "name": f"{author_username} authored comment"  # Add edge name
            }
        )
        
        yield authored_edge
        
        # Create comment-PR relationship
        comment_pr_edge = Edge(
            source_node_id=str(comment.id),
            target_node_id=str(pr.id),
            relationship_name=RelationshipType.BELONGS_TO.value,
            properties={
                "weight": 1.0,
                "name": f"Comment belongs to PR #{pr.number}"  # Add edge name
            }
        )
        
        yield comment_pr_edge
        
        # Create interaction between PR author and comment author
        if author_id != pr.creator_id:
            # Create the two-way interaction
            interaction_id = uuid5(NAMESPACE_OID, f"{pr.id}_{comment_id}_{author_id}_{pr.creator_id}")
            sentiment = analyze_comment_sentiment(comment.body)
            
            # Calculate response time if possible
            response_time = None
            try:
                pr_created = datetime.fromisoformat(pr.created_at.replace('Z', '+00:00'))
                comment_created = datetime.fromisoformat(comment.created_at.replace('Z', '+00:00'))
                response_time = int((comment_created - pr_created).total_seconds())
            except (ValueError, AttributeError, TypeError):
                pass
            
            interaction = DeveloperInteraction(
                id=interaction_id,
                interaction_type="review" if "changes_requested" in comment.body.lower() else "comment",
                developer1_id=author_id,
                developer2_id=pr.creator_id,
                developer1_username=author_username,
                developer2_username=pr.creator_username,
                sentiment=sentiment,
                response_time_seconds=response_time,
                pull_request_id=str(pr.id),
                repository_id=str(repo.id),
                comment_id=str(comment_id),
                date=comment.created_at
            )
            
            yield interaction
            
            # Create interaction edge
            interaction_edge = Edge(
                source_node_id=str(developers[author_id].id),
                target_node_id=str(developers[pr.creator_id].id),
                relationship_name=RelationshipType.INTERACTED_WITH.value,
                properties={
                    "weight": get_sentiment_weight(sentiment),
                    "sentiment": sentiment,
                    "interaction_type": interaction.interaction_type,
                    "name": f"{author_username} interacted with {pr.creator_username} ({sentiment})"  # Add edge name
                }
            )
            
            yield interaction_edge


def analyze_comment_sentiment(text: str) -> str:
    """
    Simple rule-based sentiment analysis for PR comments.
    
    Args:
        text: Comment text
    
    Returns:
        Sentiment label: 'positive', 'negative', or 'neutral'
    """
    text = text.lower()
    
    # Simple positive/negative word lists
    positive_words = [
        "good", "great", "excellent", "nice", "thanks", "lgtm", "approve", 
        "awesome", "well done", "impressive", "thanks", "thank you", "wonderful",
        "perfect", "+1", "👍", ":+1:", "😄", "🎉"
    ]
    
    negative_words = [
        "bad", "issue", "error", "problem", "fix", "wrong", "bug", "incorrect",
        "needs improvement", "needs work", "not working", "doesn't work", 
        "change", "fail", "reject", "rejected", "changes requested", "-1", "👎",
        ":-1:", "😞", "🐛"
    ]
    
    # Count occurrences
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Determine sentiment
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


def get_sentiment_weight(sentiment: str) -> float:
    """
    Convert sentiment to edge weight.
    
    Args:
        sentiment: Sentiment label
    
    Returns:
        Weight value
    """
    if sentiment == "positive":
        return 0.8
    elif sentiment == "negative":
        return 0.3
    else:
        return 0.5 