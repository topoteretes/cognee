"""
Perform temporal analysis of GitHub developer activities and relationships.
"""
import logging
from typing import AsyncGenerator, Dict, List, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict
from uuid import uuid4, NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.shared.data_models import Node, Edge, RelationshipType
from cognee.tasks.github.fetch_repositories import Repository, Developer
from cognee.tasks.github.pull_requests import PullRequest, PRComment, DeveloperInteraction

logger = logging.getLogger(__name__)

class TemporalActivity(DataPoint):
    """Entity representing a developer's activity over time."""
    
    class Meta:
        category = "github"
        index_fields = ["developer_username", "time_period"]
    
    def __init__(
        self,
        id,
        developer_id: str,
        developer_username: str,
        time_period: str,  # e.g., "2023-Q1", "2022-07", etc.
        start_date: str,
        end_date: str,
        commit_count: int = 0,
        pr_count: int = 0,
        review_count: int = 0,
        comment_count: int = 0,
        repositories: List[str] = None,
        **kwargs
    ):
        super().__init__(id=id)
        self.developer_id = developer_id
        self.developer_username = developer_username
        self.time_period = time_period
        self.start_date = start_date
        self.end_date = end_date
        self.commit_count = commit_count
        self.pr_count = pr_count
        self.review_count = review_count
        self.comment_count = comment_count
        self.repositories = repositories or []
        
        # Store all other metadata
        for key, value in kwargs.items():
            setattr(self, key, value)


class DeveloperTrajectory(DataPoint):
    """Entity representing a developer's career trajectory."""
    
    class Meta:
        category = "github"
        index_fields = ["developer_username"]
    
    def __init__(
        self,
        id,
        developer_id: str,
        developer_username: str,
        career_start: str = None,
        companies: List[Tuple[str, str, str]] = None,  # List of (company, start_date, end_date)
        technology_timeline: Dict[str, List[str]] = None,  # Technology -> list of time periods
        collaborator_count_history: Dict[str, int] = None,  # Time period -> collaborator count
        **kwargs
    ):
        super().__init__(id=id)
        self.developer_id = developer_id
        self.developer_username = developer_username
        self.career_start = career_start
        self.companies = companies or []
        self.technology_timeline = technology_timeline or {}
        self.collaborator_count_history = collaborator_count_history or {}
        
        # Store all other metadata
        for key, value in kwargs.items():
            setattr(self, key, value)


async def analyze_temporal_activities(
    developer: Developer,
    pull_requests: List[PullRequest],
    comments: List[PRComment],
    interactions: List[DeveloperInteraction],
    contribution_dates: Dict[str, List[str]]  # Repo ID -> list of contribution dates
) -> AsyncGenerator[Union[DataPoint, TemporalActivity, Edge], None]:
    """
    Analyze a developer's activities over time.
    
    Args:
        developer: Developer object
        pull_requests: List of PullRequest objects authored by the developer
        comments: List of PRComment objects authored by the developer
        interactions: List of DeveloperInteraction objects involving the developer
        contribution_dates: Dictionary mapping repo IDs to lists of contribution dates
    """
    dev_id = str(developer.id)
    
    # Group activities by time period (quarters)
    # Format: {"2023-Q1": {"prs": [], "comments": [], "repos": set()}}
    time_periods = defaultdict(lambda: {"prs": [], "comments": [], "reviews": [], "repos": set()})
    
    # Process pull requests
    for pr in pull_requests:
        if not pr.created_at:
            continue
            
        try:
            date = datetime.fromisoformat(pr.created_at.replace("Z", "+00:00"))
            period = f"{date.year}-Q{(date.month-1)//3+1}"
            
            time_periods[period]["prs"].append(pr)
            time_periods[period]["repos"].add(pr.repository_id)
        except (ValueError, AttributeError):
            continue
    
    # Process comments
    for comment in comments:
        if not comment.created_at:
            continue
            
        try:
            date = datetime.fromisoformat(comment.created_at.replace("Z", "+00:00"))
            period = f"{date.year}-Q{(date.month-1)//3+1}"
            
            # Check if this is a review or a regular comment
            if comment.body and "changes requested" in comment.body.lower():
                time_periods[period]["reviews"].append(comment)
            else:
                time_periods[period]["comments"].append(comment)
                
            time_periods[period]["repos"].add(comment.repository_id)
        except (ValueError, AttributeError):
            continue
    
    # Process contribution dates
    for repo_id, dates in contribution_dates.items():
        for date_str in dates:
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                period = f"{date.year}-Q{(date.month-1)//3+1}"
                
                time_periods[period]["repos"].add(repo_id)
            except (ValueError, AttributeError):
                continue
    
    # Create temporal activity entities
    for period, data in time_periods.items():
        # Parse period to determine date range
        year = int(period.split("-")[0])
        quarter = int(period.split("Q")[1])
        start_month = (quarter - 1) * 3 + 1
        
        # Create date range
        start_date = datetime(year, start_month, 1).isoformat()
        if start_month + 3 > 12:
            end_date = datetime(year + 1, (start_month + 3) % 12, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, start_month + 3, 1) - timedelta(days=1)
        end_date = end_date.isoformat()
        
        # Create temporal activity entity
        activity_id = uuid5(NAMESPACE_OID, f"{dev_id}_{period}")
        
        activity = TemporalActivity(
            id=activity_id,
            developer_id=dev_id,
            developer_username=developer.username,
            time_period=period,
            start_date=start_date,
            end_date=end_date,
            pr_count=len(data["prs"]),
            review_count=len(data["reviews"]),
            comment_count=len(data["comments"]),
            repositories=list(data["repos"]),
            total_activity_count=len(data["prs"]) + len(data["reviews"]) + len(data["comments"])
        )
        
        yield activity
        
        # Create temporal activity edge
        activity_edge = Edge(
            node1=activity,
            node2=developer,
            attributes={
                "relationship_type": RelationshipType.ACTIVITY_OF.value,
                "weight": 1.0,
                "time_period": period,
            }
        )
        
        yield activity_edge


async def analyze_developer_trajectory(
    developer: Developer,
    temporal_activities: List[TemporalActivity],
    repos_by_id: Dict[str, Repository],
    collaborations_by_period: Dict[str, Set[str]]  # time_period -> set of collaborator IDs
) -> AsyncGenerator[Union[DataPoint, DeveloperTrajectory, Edge], None]:
    """
    Analyze a developer's career trajectory based on temporal data.
    
    Args:
        developer: Developer object
        temporal_activities: List of TemporalActivity objects
        repos_by_id: Dictionary mapping repo IDs to Repository objects
        collaborations_by_period: Dictionary mapping time periods to sets of collaborator IDs
    """
    dev_id = str(developer.id)
    
    # Sort activities by time period
    sorted_activities = sorted(
        temporal_activities, 
        key=lambda x: x.start_date
    )
    
    if not sorted_activities:
        return
    
    # Extract career start
    career_start = sorted_activities[0].start_date if sorted_activities else None
    
    # Extract company information from user profile and repositories
    # This is a simplified approach - in reality, you'd need more complex 
    # heuristics to determine company affiliations
    companies = []
    current_company = developer.company
    
    if current_company:
        # Start with current company from profile
        latest_activity_date = sorted_activities[-1].end_date if sorted_activities else None
        companies.append((current_company, None, latest_activity_date))
    
    # Extract technologies over time
    technology_timeline = defaultdict(set)
    
    for activity in sorted_activities:
        # Get repos for this period
        period_repos = [repos_by_id.get(repo_id) for repo_id in activity.repositories if repo_id in repos_by_id]
        
        # Extract technologies from repos
        for repo in period_repos:
            if hasattr(repo, "language") and repo.language:
                technology_timeline[repo.language].add(activity.time_period)
    
    # Convert sets to sorted lists
    technology_timeline_dict = {
        tech: sorted(periods) 
        for tech, periods in technology_timeline.items()
    }
    
    # Extract collaborator count history
    collaborator_count_history = {
        period: len(collaborator_ids)
        for period, collaborator_ids in collaborations_by_period.items()
    }
    
    # Create developer trajectory entity
    trajectory_id = uuid5(NAMESPACE_OID, f"{dev_id}_trajectory")
    
    trajectory = DeveloperTrajectory(
        id=trajectory_id,
        developer_id=dev_id,
        developer_username=developer.username,
        career_start=career_start,
        companies=companies,
        technology_timeline=technology_timeline_dict,
        collaborator_count_history=collaborator_count_history
    )
    
    yield trajectory
    
    # Create trajectory edge
    trajectory_edge = Edge(
        node1=trajectory,
        node2=developer,
        attributes={
            "relationship_type": RelationshipType.TRAJECTORY_OF.value,
            "weight": 1.0,
        }
    )
    
    yield trajectory_edge 