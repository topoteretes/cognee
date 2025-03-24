"""
Fetch repositories from GitHub for a specific user.
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Any, Optional, Set, Tuple, Union
import aiohttp
from uuid import uuid4, NAMESPACE_OID, uuid5, UUID
from pydantic import Field, field_validator

from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, RelationshipType, Edge
from cognee.tasks.github.config import GitHubSettings

logger = logging.getLogger(__name__)

class Repository(DataPoint):
    """Repository entity from GitHub."""
    
    class Meta:
        category = "github"
        index_fields = ["name", "description"]
    
    name: str
    description: Optional[str] = None
    language: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    url: Optional[str] = None
    html_url: Optional[str] = None
    api_url: Optional[str] = None
    owner: Optional[str] = None
    fork: Optional[bool] = None
    forks_count: Optional[int] = None
    stargazers_count: Optional[int] = None
    watchers_count: Optional[int] = None
    open_issues_count: Optional[int] = None
    default_branch: Optional[str] = None
    
    def __init__(
        self,
        id: UUID,
        name: str,
        description: Optional[str] = None,
        language: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        url: Optional[str] = None,
        html_url: Optional[str] = None,
        api_url: Optional[str] = None,
        owner: Optional[str] = None,
        fork: Optional[bool] = None,
        forks_count: Optional[int] = None,
        stargazers_count: Optional[int] = None,
        watchers_count: Optional[int] = None,
        open_issues_count: Optional[int] = None,
        default_branch: Optional[str] = None,
        **kwargs
    ):
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "name": name,
            "description": description,
            "language": language,
            "created_at": created_at,
            "updated_at": updated_at,
            "url": url,
            "html_url": html_url,
            "api_url": api_url,
            "owner": owner,
            "fork": fork,
            "forks_count": forks_count,
            "stargazers_count": stargazers_count,
            "watchers_count": watchers_count,
            "open_issues_count": open_issues_count,
            "default_branch": default_branch,
            **kwargs
        }
        super().__init__(**data)


class Developer(DataPoint):
    """Developer entity from GitHub."""
    
    class Meta:
        category = "github"
        index_fields = ["username", "name"]
    
    username: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    html_url: Optional[str] = None
    api_url: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    created_at: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    
    # Collaboration network metrics
    collaboration_count: Optional[int] = 0
    collaboration_repo_count: Optional[int] = 0
    degree_centrality: Optional[float] = 0.0
    
    def __init__(
        self,
        id: UUID,
        username: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        avatar_url: Optional[str] = None,
        html_url: Optional[str] = None,
        api_url: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        bio: Optional[str] = None,
        created_at: Optional[str] = None,
        public_repos: Optional[int] = None,
        followers: Optional[int] = None,
        following: Optional[int] = None,
        collaboration_count: Optional[int] = 0,
        collaboration_repo_count: Optional[int] = 0,
        degree_centrality: Optional[float] = 0.0,
        **kwargs
    ):
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "username": username,
            "name": name if name else username,  # Use username as fallback for name
            "email": email,
            "avatar_url": avatar_url,
            "html_url": html_url,
            "api_url": api_url,
            "company": company,
            "location": location,
            "bio": bio,
            "created_at": created_at,
            "public_repos": public_repos,
            "followers": followers,
            "following": following,
            "collaboration_count": collaboration_count,
            "collaboration_repo_count": collaboration_repo_count,
            "degree_centrality": degree_centrality,
            **kwargs
        }
        super().__init__(**data)


async def fetch_github_data(url: str, headers: Dict = None) -> Dict:
    """Fetch data from GitHub API with rate limit handling."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 403 and 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
                reset_time = int(response.headers['X-RateLimit-Reset'])
                current_time = int(time.time())
                sleep_time = reset_time - current_time + 5  # Add 5 seconds buffer
                
                if sleep_time > 0:
                    logger.warning(f"GitHub API rate limit reached. Sleeping for {sleep_time} seconds.")
                    await asyncio.sleep(sleep_time)
                    # Retry the request
                    return await fetch_github_data(url, headers)
            
            if response.status != 200:
                logger.error(f"GitHub API error: {await response.text()}")
                return {}
            
            return await response.json()


async def fetch_repositories(username: str, api_token: str = None, max_repos: int = 5) -> AsyncGenerator[DataPoint, None]:
    """
    Fetch repositories for a given GitHub username, limited to a maximum number.
    
    Args:
        username: GitHub username
        api_token: Optional GitHub API token for authentication
        max_repos: Maximum number of repositories to fetch (default: 5)
    """
    headers = {}
    if api_token:
        headers["Authorization"] = f"token {api_token}"
    
    # First yield the developer
    developer_url = f"https://api.github.com/users/{username}"
    developer_data = await fetch_github_data(developer_url, headers)
    
    if not developer_data:
        logger.error(f"Could not fetch developer data for {username}")
        return
    
    # Create the developer with required fields
    developer = Developer(
        id=uuid5(NAMESPACE_OID, str(developer_data.get("id", ""))),
        username=username,  # Important: Use the username passed to the function, not the API result
        name=developer_data.get("name"),
        email=developer_data.get("email"),
        avatar_url=developer_data.get("avatar_url"),
        html_url=developer_data.get("html_url"),
        api_url=developer_data.get("url"),
        company=developer_data.get("company"),
        location=developer_data.get("location"),
        bio=developer_data.get("bio"),
        created_at=developer_data.get("created_at"),
        public_repos=developer_data.get("public_repos"),
        followers=developer_data.get("followers"),
        following=developer_data.get("following")
    )
    
    yield developer
    
    # Now fetch limited number of repositories
    repos_url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page={max_repos}"
    repos_data = await fetch_github_data(repos_url, headers)
    
    if not repos_data:
        logger.warning(f"No repositories found for user {username}")
        return
    
    # Limit to max_repos
    for repo_data in repos_data[:max_repos]:
        repo_id = uuid5(NAMESPACE_OID, str(repo_data.get("id", "")))
        repo_name = repo_data.get("name", "")
        
        # Create a human-readable display name for the repository
        display_name = f"{username}/{repo_name}"
        
        repo = Repository(
            id=repo_id,
            name=display_name,  # Use owner/name format for better readability
            description=repo_data.get("description"),
            language=repo_data.get("language"),
            created_at=repo_data.get("created_at"),
            updated_at=repo_data.get("updated_at"),
            url=repo_data.get("url"),
            html_url=repo_data.get("html_url"),
            api_url=repo_data.get("url"),
            owner=repo_data["owner"].get("login"),
            fork=repo_data.get("fork"),
            forks_count=repo_data.get("forks_count"),
            stargazers_count=repo_data.get("stargazers_count"),
            watchers_count=repo_data.get("watchers_count"),
            open_issues_count=repo_data.get("open_issues_count"),
            default_branch=repo_data.get("default_branch"),
        )
        
        yield repo
        
        # Create an ownership relationship
        owner_edge = Edge(
            source_node_id=str(developer.id),
            target_node_id=str(repo_id),
            relationship_name=RelationshipType.OWNS.value,
            properties={
                "weight": 1.0,
            }
        )
        
        yield owner_edge 