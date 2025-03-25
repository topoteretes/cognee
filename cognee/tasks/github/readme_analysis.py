"""
Fetch and analyze README files from GitHub repositories.
"""
import logging
import base64
from typing import AsyncGenerator, Dict, List, Any, Optional
from uuid import uuid4, NAMESPACE_OID, uuid5, UUID

from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, Edge, RelationshipType
from cognee.tasks.github.fetch_repositories import Repository, Developer, fetch_github_data

logger = logging.getLogger(__name__)

class ReadmeDocument(DataPoint):
    """README document entity from GitHub."""
    
    class Meta:
        category = "github"
        index_fields = ["content", "repo_name"]
    
    content: str
    repo_id: str
    repo_name: str
    repo_owner: str
    topics: List[str]
    technologies: List[str]
    url: Optional[str] = None
    html_url: Optional[str] = None
    api_url: Optional[str] = None
    
    def __init__(
        self,
        id: UUID,
        content: str,
        repo_id: str,
        repo_name: str,
        repo_owner: str,
        topics: List[str] = None,
        technologies: List[str] = None,
        url: Optional[str] = None,
        html_url: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs
    ):
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "content": content,
            "repo_id": repo_id,
            "repo_name": repo_name,
            "repo_owner": repo_owner,
            "topics": topics or [],
            "technologies": technologies or [],
            "url": url,
            "html_url": html_url,
            "api_url": api_url,
            **kwargs
        }
        super().__init__(**data)


async def fetch_readme(repo: Repository, api_token: str = None) -> AsyncGenerator[DataPoint, None]:
    """
    Fetch README files for a given repository.
    
    Args:
        repo: Repository object
        api_token: GitHub API token for authentication
    """
    headers = {}
    if api_token:
        headers["Authorization"] = f"token {api_token}"
    
    # Fetch README content
    readme_url = f"https://api.github.com/repos/{repo.owner}/{repo.name}/readme"
    readme_data = await fetch_github_data(readme_url, headers)
    
    if not readme_data:
        logger.warning(f"No README found for repository {repo.name}")
        return
    
    # Decode content
    content = ""
    if "content" in readme_data:
        try:
            content = base64.b64decode(readme_data["content"]).decode("utf-8")
        except Exception as e:
            logger.error(f"Error decoding README content: {e}")
            return
    
    # Also fetch repository topics
    topics_url = f"https://api.github.com/repos/{repo.owner}/{repo.name}/topics"
    custom_headers = headers.copy()
    custom_headers["Accept"] = "application/vnd.github.mercy-preview+json"  # Required for topics API
    
    topics_data = await fetch_github_data(topics_url, custom_headers)
    topics = topics_data.get("names", []) if topics_data else []
    
    # Extract technologies from content
    technologies = extract_technologies(content)
    
    # Create README entity
    readme_id = uuid5(NAMESPACE_OID, f"{repo.id}_readme")
    
    readme = ReadmeDocument(
        id=readme_id,
        content=content,
        repo_id=str(repo.id),
        repo_name=repo.name,
        repo_owner=repo.owner,
        topics=topics,
        technologies=technologies,
        url=readme_data.get("url"),
        html_url=readme_data.get("html_url"),
        api_url=readme_data.get("url")
    )
    
    yield readme
    
    # Create belongs_to relationship to repository
    belongs_to_edge = Edge(
        source_node_id=str(readme.id),
        target_node_id=str(repo.id),
        relationship_name=RelationshipType.BELONGS_TO.value,
        properties={
            "weight": 1.0,
        }
    )
    
    yield belongs_to_edge
    
    # Create technology relationships
    for tech in technologies:
        # Create a technology entity ID
        tech_id = uuid5(NAMESPACE_OID, tech.lower())
        
        # Create technology relationship
        tech_edge = Edge(
            source_node_id=str(repo.id),
            target_node_id=str(tech_id),
            relationship_name=RelationshipType.USES_TECHNOLOGY.value,
            properties={
                "weight": 0.8,
                "technology": tech,
            }
        )
        
        yield tech_edge


def extract_technologies(content: str) -> List[str]:
    """
    Extract technologies mentioned in the README content.
    This is a simple approach - in a real implementation, 
    you might use more sophisticated NLP techniques.
    
    Args:
        content: README content
    
    Returns:
        List of detected technologies
    """
    # List of common technologies to detect
    technology_keywords = [
        # Programming languages
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
        # Frameworks
        "React", "Angular", "Vue", "Django", "Flask", "Express", "Spring", "ASP.NET", "Rails",
        # Libraries and tools
        "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "jQuery", "Bootstrap",
        # Databases
        "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Redis", "Elasticsearch",
        # Cloud providers and services
        "AWS", "Azure", "Google Cloud", "Firebase", "Heroku",
        # DevOps tools
        "Docker", "Kubernetes", "Jenkins", "GitHub Actions", "Travis CI", "CircleCI",
        # Mobile
        "iOS", "Android", "React Native", "Flutter", "Swift", "Kotlin"
    ]
    
    # Check for presence of each technology
    found_technologies = []
    content_lower = content.lower()
    
    for tech in technology_keywords:
        if tech.lower() in content_lower:
            found_technologies.append(tech)
    
    return found_technologies 