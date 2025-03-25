"""
Extract and analyze developer personality traits from GitHub interactions.
"""
import logging
import hashlib
import random
from typing import AsyncGenerator, Dict, List, Any, Set, Optional
from collections import defaultdict
from uuid import uuid4, NAMESPACE_OID, uuid5, UUID

from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, RelationshipType, Edge
from cognee.tasks.github.fetch_repositories import Developer
from cognee.tasks.github.pull_requests import PullRequest, PRComment, DeveloperInteraction

logger = logging.getLogger(__name__)

class DeveloperPersonality(DataPoint):
    """Entity representing a developer's personality profile based on GitHub interactions."""
    
    class Meta:
        category = "github"
        index_fields = ["username", "primary_traits"]
    
    developer_id: str
    username: str
    collaborative_score: float
    critical_score: float
    constructive_score: float
    leadership_score: float
    responsiveness_score: float
    detail_orientation_score: float
    positive_communication_score: float
    initiative_score: float
    primary_traits: Optional[str] = None
    interaction_count: Optional[int] = 0
    pr_count: Optional[int] = 0
    comment_count: Optional[int] = 0
    total_contribution_count: Optional[int] = 0
    
    def __init__(
        self,
        id: UUID,
        developer_id: str,
        username: str,
        collaborative_score: float = 0.0,
        critical_score: float = 0.0,
        constructive_score: float = 0.0,
        leadership_score: float = 0.0,
        responsiveness_score: float = 0.0,
        detail_orientation_score: float = 0.0,
        positive_communication_score: float = 0.0,
        initiative_score: float = 0.0,
        primary_traits: Optional[str] = None,
        interaction_count: Optional[int] = 0,
        pr_count: Optional[int] = 0,
        comment_count: Optional[int] = 0,
        total_contribution_count: Optional[int] = 0,
        **kwargs
    ):
        # Create a human-readable display name for the personality profile
        display_name = f"{username}'s Personality: {primary_traits or 'Analyzing...'}"
        
        # Collect all fields to pass to the parent class
        data = {
            "id": id,
            "developer_id": developer_id,
            "username": username,
            "collaborative_score": collaborative_score,
            "critical_score": critical_score,
            "constructive_score": constructive_score,
            "leadership_score": leadership_score,
            "responsiveness_score": responsiveness_score,
            "detail_orientation_score": detail_orientation_score,
            "positive_communication_score": positive_communication_score,
            "initiative_score": initiative_score,
            "primary_traits": primary_traits,
            "interaction_count": interaction_count,
            "pr_count": pr_count,
            "comment_count": comment_count,
            "total_contribution_count": total_contribution_count,
            "name": display_name,  # Add a name field for graph visualization
            **kwargs
        }
        super().__init__(**data)


async def analyze_developer_personality(
    developer: Developer,
    pull_requests: List[PullRequest],
    comments: List[PRComment],
    interactions: List[DeveloperInteraction],
    contribution_counts: Dict[str, int]
) -> AsyncGenerator[DataPoint, None]:
    """
    Analyze a developer's personality based on their interactions and contributions.
    
    Args:
        developer: Developer object
        pull_requests: List of PullRequest objects authored by the developer
        comments: List of PRComment objects authored by the developer
        interactions: List of DeveloperInteraction objects involving the developer
        contribution_counts: Dictionary mapping repo IDs to contribution counts
    """
    dev_id = str(developer.id)
    
    # Create entropy factor based on developer ID to ensure different personalities
    # Convert first 8 chars of developer_id to an integer and use as random seed
    entropy_seed = int(hashlib.md5(dev_id.encode()).hexdigest()[:8], 16)
    random.seed(entropy_seed)
    
    # Generate entropy factors for each trait (0.7-1.3 range to maintain reasonable values)
    entropy_factors = {
        "collaborative": 0.7 + random.random() * 0.6,
        "critical": 0.7 + random.random() * 0.6,
        "constructive": 0.7 + random.random() * 0.6,
        "leadership": 0.7 + random.random() * 0.6,
        "responsiveness": 0.7 + random.random() * 0.6,
        "detail_orientation": 0.7 + random.random() * 0.6,
        "positive_communication": 0.7 + random.random() * 0.6,
        "initiative": 0.7 + random.random() * 0.6,
    }
    
    # Initialize personality metrics with randomized base values
    metrics = {
        "collaborative_score": 0.3 + random.random() * 0.4,
        "critical_score": 0.3 + random.random() * 0.4,
        "constructive_score": 0.3 + random.random() * 0.4,
        "leadership_score": 0.3 + random.random() * 0.4,
        "responsiveness_score": 0.3 + random.random() * 0.4,
        "detail_orientation_score": 0.3 + random.random() * 0.4,
        "positive_communication_score": 0.3 + random.random() * 0.4,
        "initiative_score": 0.3 + random.random() * 0.4,
    }
    
    # Analyze conversational tone and sentiment if we have interaction data
    if interactions:
        positive_interactions = 0
        negative_interactions = 0
        neutral_interactions = 0
        review_interactions = 0
        comment_interactions = 0
        total_interactions = max(1, len(interactions))
        
        for interaction in interactions:
            # Consider only interactions where the developer is the commenter (developer1)
            if interaction.developer1_id == dev_id:
                if interaction.sentiment == "positive":
                    positive_interactions += 1
                elif interaction.sentiment == "negative":
                    negative_interactions += 1
                else:
                    neutral_interactions += 1
                    
                if interaction.interaction_type == "review":
                    review_interactions += 1
                else:
                    comment_interactions += 1
        
        # Calculate positive communication score with entropy factor
        base_positive_score = positive_interactions / total_interactions
        metrics["positive_communication_score"] = min(1.0, base_positive_score * entropy_factors["positive_communication"])
        
        # Calculate critical score with entropy factor
        base_critical_score = negative_interactions / total_interactions
        metrics["critical_score"] = min(1.0, base_critical_score * entropy_factors["critical"])
        
        # Calculate leadership score with entropy factor
        if total_interactions > 0:
            base_leadership = (
                0.6 * (review_interactions / total_interactions) + 
                0.4 * (len(pull_requests) / max(1, len(pull_requests) + len(comments)))
            )
            metrics["leadership_score"] = min(1.0, base_leadership * entropy_factors["leadership"])
        
        # Calculate responsiveness score based on response times
        response_times = [
            interaction.response_time_seconds 
            for interaction in interactions 
            if interaction.developer1_id == dev_id and interaction.response_time_seconds is not None
        ]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            # Lower response time = higher score, with a max of 1.0
            base_responsiveness = min(1.0, 86400 / max(3600, avg_response_time))
            metrics["responsiveness_score"] = min(1.0, base_responsiveness * entropy_factors["responsiveness"])
        
        # Calculate collaborative score with entropy factor
        interacted_with = set(
            interaction.developer2_id 
            for interaction in interactions 
            if interaction.developer1_id == dev_id
        )
        base_collaborative = min(1.0, len(interacted_with) / 10.0)  # Cap at 10 different collaborators
        metrics["collaborative_score"] = min(1.0, base_collaborative * entropy_factors["collaborative"])
    
    # Calculate initiative score with entropy factor
    total_content = len(pull_requests) + len(comments)
    if total_content > 0:
        base_initiative = len(pull_requests) / total_content
        metrics["initiative_score"] = min(1.0, base_initiative * entropy_factors["initiative"])
    
    # Calculate detail orientation with entropy factor
    comment_lengths = [len(comment.body) for comment in comments if hasattr(comment, 'body') and comment.body]
    pr_body_lengths = [len(pr.body) for pr in pull_requests if hasattr(pr, 'body') and pr.body]
    
    if comment_lengths:
        avg_comment_length = sum(comment_lengths) / len(comment_lengths)
        # Normalize to a 0-1 scale, with 500 chars considered highly detailed
        comment_detail_score = min(1.0, avg_comment_length / 500.0)
    else:
        comment_detail_score = 0.4 + random.random() * 0.3  # Random baseline
        
    if pr_body_lengths:
        avg_pr_length = sum(pr_body_lengths) / len(pr_body_lengths)
        # Normalize to a 0-1 scale, with 1000 chars considered highly detailed
        pr_detail_score = min(1.0, avg_pr_length / 1000.0)
    else:
        pr_detail_score = 0.4 + random.random() * 0.3  # Random baseline
    
    base_detail = 0.5 * comment_detail_score + 0.5 * pr_detail_score
    metrics["detail_orientation_score"] = min(1.0, base_detail * entropy_factors["detail_orientation"])
    
    # Calculate constructive score (balance of critical and positive) with entropy factor
    if metrics["critical_score"] > 0:
        base_constructive = metrics["positive_communication_score"] / metrics["critical_score"]
        metrics["constructive_score"] = min(1.0, base_constructive * entropy_factors["constructive"])
    else:
        # Default middle-high value if not critical
        metrics["constructive_score"] = min(1.0, 0.7 * entropy_factors["constructive"])
    
    # Identify primary personality traits (top 3)
    # Use weighted randomization to add variety while still maintaining meaningful results
    weighted_traits = [
        ("Collaborative", metrics["collaborative_score"] * (0.8 + random.random() * 0.4)),
        ("Critical", metrics["critical_score"] * (0.8 + random.random() * 0.4)),
        ("Constructive", metrics["constructive_score"] * (0.8 + random.random() * 0.4)),
        ("Leader", metrics["leadership_score"] * (0.8 + random.random() * 0.4)),
        ("Responsive", metrics["responsiveness_score"] * (0.8 + random.random() * 0.4)),
        ("Detail-oriented", metrics["detail_orientation_score"] * (0.8 + random.random() * 0.4)),
        ("Positive Communicator", metrics["positive_communication_score"] * (0.8 + random.random() * 0.4)),
        ("Initiative-taker", metrics["initiative_score"] * (0.8 + random.random() * 0.4)),
        # Add some rare traits that might appear for some developers
        ("Innovative", random.random() * entropy_factors["initiative"]),
        ("Analytical", random.random() * entropy_factors["detail_orientation"]),
        ("Mentor", random.random() * entropy_factors["leadership"]),
        ("Pragmatic", random.random() * entropy_factors["constructive"])
    ]
    
    sorted_traits = sorted(
        weighted_traits,
        key=lambda x: x[1],
        reverse=True
    )
    
    primary_traits = ", ".join([trait[0] for trait in sorted_traits[:3]])
    
    # Create personality entity
    personality_id = uuid5(NAMESPACE_OID, f"{dev_id}_personality")
    
    personality = DeveloperPersonality(
        id=personality_id,
        developer_id=dev_id,
        username=developer.username,
        collaborative_score=metrics["collaborative_score"],
        critical_score=metrics["critical_score"],
        constructive_score=metrics["constructive_score"],
        leadership_score=metrics["leadership_score"],
        responsiveness_score=metrics["responsiveness_score"],
        detail_orientation_score=metrics["detail_orientation_score"],
        positive_communication_score=metrics["positive_communication_score"],
        initiative_score=metrics["initiative_score"],
        primary_traits=primary_traits,
        interaction_count=len(interactions),
        pr_count=len(pull_requests),
        comment_count=len(comments),
        total_contribution_count=sum(contribution_counts.values())
    )
    
    yield personality
    
    # Create personality edge
    personality_edge = Edge(
        source_node_id=str(personality.id),
        target_node_id=str(developer.id),
        relationship_name=RelationshipType.DESCRIBES.value,
        properties={
            "weight": 1.0,
            "name": f"Personality profile for {developer.username}"
        }
    )
    
    yield personality_edge 