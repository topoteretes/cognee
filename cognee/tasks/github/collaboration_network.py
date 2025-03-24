"""
Build and analyze the collaboration network from GitHub repositories and contributors.
"""
import logging
import itertools
from typing import AsyncGenerator, Dict, List, Set, Any, Union, Optional, Tuple
from uuid import uuid4, NAMESPACE_OID, uuid5, UUID

from pydantic import Field
from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, Edge, RelationshipType
from cognee.tasks.github.fetch_repositories import Repository, Developer

logger = logging.getLogger(__name__)

class Collaboration(DataPoint):
    """
    Represents a collaboration between two developers on one or more repositories.
    """
    developer1_id: str = Field(..., description="ID of the first developer")
    developer2_id: str = Field(..., description="ID of the second developer")
    developer1_username: str = Field(..., description="Username of the first developer")
    developer2_username: str = Field(..., description="Username of the second developer")
    repository_count: int = Field(..., description="Number of repositories collaborated on")
    repositories: List[str] = Field(default_factory=list, description="List of repository names")
    
    def __init__(
        self,
        id: UUID,
        developer1_id: str,
        developer2_id: str,
        developer1_username: str,
        developer2_username: str,
        repository_count: int,
        repositories: List[str] = None,
        **kwargs
    ):
        # Create a human-readable display name for the collaboration
        display_name = f"Collaboration: {developer1_username} & {developer2_username} ({repository_count} repos)"
        
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "developer1_id": developer1_id,
            "developer2_id": developer2_id,
            "developer1_username": developer1_username,
            "developer2_username": developer2_username,
            "repository_count": repository_count,
            "repositories": repositories or [],
            "name": display_name,  # Add a name field for graph visualization
            **kwargs
        }
        super().__init__(**data)


async def build_collaboration_network(
    repos: List[Repository],
    developers: Dict[str, Developer],
    repo_contributors: Dict[str, List[str]]
) -> AsyncGenerator[Union[DataPoint, Collaboration, Edge], None]:
    """
    Build a collaboration network based on shared repository contributions.
    
    Args:
        repos: List of Repository objects
        developers: Dictionary mapping developer IDs to Developer objects
        repo_contributors: Dictionary mapping repository IDs to lists of contributor IDs
    """
    # Dictionary to track collaborations between developers
    # (dev1_id, dev2_id) -> [repo_ids]
    collaborations = {}
    
    # Process each repository to find collaborations
    for repo in repos:
        repo_id = str(repo.id)
        if repo_id not in repo_contributors:
            continue
            
        # Get all contributors for this repository
        contributors = repo_contributors[repo_id]
        
        # For each pair of contributors, record their collaboration
        for dev1_id, dev2_id in itertools.combinations(contributors, 2):
            collab_key = tuple(sorted([dev1_id, dev2_id]))
            
            if collab_key not in collaborations:
                collaborations[collab_key] = []
                
            collaborations[collab_key].append(repo_id)
    
    # Create collaboration entities and relationships
    for (dev1_id, dev2_id), repo_ids in collaborations.items():
        dev1 = developers.get(dev1_id)
        dev2 = developers.get(dev2_id)
        
        if not dev1 or not dev2:
            continue
        
        collab_id = uuid5(NAMESPACE_OID, f"{dev1_id}_{dev2_id}")
        repo_count = len(repo_ids)
        
        # Get repository names
        repo_names = [repo.name for repo in repos if str(repo.id) in repo_ids]
        
        # Create collaboration entity
        collaboration = Collaboration(
            id=collab_id,
            developer1_id=dev1_id,
            developer2_id=dev2_id,
            developer1_username=dev1.username,
            developer2_username=dev2.username,
            repository_count=repo_count,
            repositories=repo_names
        )
        
        yield collaboration
        
        # Create collaboration edge
        collab_edge = Edge(
            source_node_id=str(dev1.id),
            target_node_id=str(dev2.id),
            relationship_name=RelationshipType.COLLABORATED_WITH.value,
            properties={
                "weight": min(1.0, float(repo_count) / 5.0),  # Normalize weight, max at 5 repos
                "repository_count": repo_count,
                "repositories": repo_names,
                "name": f"{dev1.username} collaborated with {dev2.username} on {repo_count} repos"  # Add edge name
            }
        )
        
        yield collab_edge


async def analyze_collaboration_network(
    developers: Dict[str, Developer],
    collaborations: List[Collaboration]
) -> AsyncGenerator[Union[DataPoint, Developer], None]:
    """
    Analyze the collaboration network to identify key metrics for each developer.
    
    Args:
        developers: Dictionary mapping developer IDs to Developer objects
        collaborations: List of Collaboration objects
    """
    # Calculate basic network metrics for each developer
    dev_metrics = {dev_id: {"collab_count": 0, "repositories": set()} for dev_id in developers}
    
    # Build adjacency lists for centrality calculations
    adjacency = {dev_id: [] for dev_id in developers}
    
    for collab in collaborations:
        dev1_id = collab.developer1_id
        dev2_id = collab.developer2_id
        
        # Update direct collaboration counts
        dev_metrics[dev1_id]["collab_count"] += 1
        dev_metrics[dev2_id]["collab_count"] += 1
        
        # Update repositories
        for repo in collab.repositories:
            dev_metrics[dev1_id]["repositories"].add(repo)
            dev_metrics[dev2_id]["repositories"].add(repo)
        
        # Build adjacency lists for network analysis
        adjacency[dev1_id].append(dev2_id)
        adjacency[dev2_id].append(dev1_id)
    
    # Calculate basic centrality metrics
    for dev_id, dev in developers.items():
        metrics = dev_metrics.get(dev_id, {})
        collab_count = metrics.get("collab_count", 0)
        repo_count = len(metrics.get("repositories", set()))
        
        # Update developer with network metrics
        dev.collaboration_count = collab_count
        dev.collaboration_repo_count = repo_count
        dev.degree_centrality = collab_count / max(1, len(developers) - 1)
        
        # More advanced metrics could be added here
        
        yield dev 