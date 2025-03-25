"""
Fetch contributors for GitHub repositories.
"""
import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Any, Optional
import aiohttp
from uuid import uuid4, NAMESPACE_OID, uuid5, UUID

from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, Edge, RelationshipType
from cognee.tasks.github.fetch_repositories import fetch_github_data, Repository, Developer

logger = logging.getLogger(__name__)

class Contribution(DataPoint):
    """Contribution entity representing a developer's contributions to a repository."""
    
    class Meta:
        category = "github"
        index_fields = ["repository_name", "developer_username"]
    
    repository_id: UUID
    developer_id: UUID
    repository_name: str
    developer_username: str
    contributions_count: int
    
    def __init__(
        self,
        id: UUID,
        repository_id: UUID,
        developer_id: UUID,
        repository_name: str,
        developer_username: str,
        contributions_count: int,
        **kwargs
    ):
        # Prepare a human-readable name for display in the graph
        display_name = f"{developer_username} → {repository_name} ({contributions_count} contributions)"
        
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "repository_id": repository_id,
            "developer_id": developer_id,
            "repository_name": repository_name,
            "developer_username": developer_username,
            "contributions_count": contributions_count,
            "name": display_name,  # Add a name field for graph visualization
            **kwargs
        }
        super().__init__(**data)


async def fetch_contributors(repo: Repository, api_token: str = None, max_contributors: int = 10) -> AsyncGenerator[DataPoint, None]:
    """
    Fetch contributors for a given repository, limited to the top N contributors.
    
    Args:
        repo: Repository object
        api_token: GitHub API token for authentication
        max_contributors: Maximum number of contributors to fetch (default: 10)
    """
    headers = {}
    if api_token:
        headers["Authorization"] = f"token {api_token}"
    
    # Extract just the repo name from the full repo.name which might be in "owner/name" format
    repo_name_parts = repo.name.split('/')
    repo_name = repo_name_parts[-1] if len(repo_name_parts) > 1 else repo.name
    owner = repo.owner or repo_name_parts[0] if len(repo_name_parts) > 1 else "unknown"
    
    contributors_url = f"https://api.github.com/repos/{owner}/{repo_name}/contributors?per_page={max_contributors}"
    contributors_data = await fetch_github_data(contributors_url, headers)
    
    if not contributors_data:
        logger.warning(f"No contributors found for repository {repo.name}")
        return
    
    # Only process up to max_contributors
    for contributor_data in contributors_data[:max_contributors]:
        # Skip bots/apps
        if contributor_data.get("type") == "Bot":
            continue
            
        contributor_id = uuid5(NAMESPACE_OID, str(contributor_data.get("id", "")))
        username = contributor_data.get("login")
        
        # Fetch more detailed contributor info
        developer_url = f"https://api.github.com/users/{username}"
        developer_data = await fetch_github_data(developer_url, headers)
        
        # Create a human-readable display name for the developer
        display_name = developer_data.get("name") or username
        
        developer = Developer(
            id=contributor_id,
            username=username,
            name=display_name,  # Ensure we use a human-readable name
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
        
        # Create contribution entity
        contribution_id = uuid5(NAMESPACE_OID, f"{repo.id}_{contributor_id}")
        contributions_count = contributor_data.get("contributions", 0)
        
        contribution = Contribution(
            id=contribution_id,
            repository_id=repo.id,
            developer_id=contributor_id,
            repository_name=repo.name,
            developer_username=username,
            contributions_count=contributions_count
        )
        
        yield contribution
        
        # Create contribution relationship using the correct Edge parameters
        edge_name = f"{username} contributed to {repo_name}"
        contribution_edge = Edge(
            source_node_id=str(developer.id),
            target_node_id=str(repo.id),
            relationship_name=RelationshipType.CONTRIBUTED_TO.value,
            properties={
                "weight": float(contributions_count) / 100.0 if contributions_count > 0 else 0.1,  # Normalize weight
                "contributions_count": contributions_count,
                "name": edge_name,  # Add a name for the edge in visualization
            }
        )
        
        yield contribution_edge