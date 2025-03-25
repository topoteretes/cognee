"""
GitHub Developer Analysis Pipeline

This pipeline analyzes a GitHub developer's repositories, contributions, and interactions
to create a comprehensive knowledge graph of their work history, collaborations,
and personality traits.
"""
import asyncio
import logging
from uuid import NAMESPACE_OID, uuid5

from typing import Dict, List, Set, AsyncGenerator, Any, Optional

from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, MonitoringTool, DataPoint, Edge
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.modules.data.deletion import prune_data, prune_system
from cognee.infrastructure.llm.methods import get_llm_response

from cognee.tasks.github.fetch_repositories import fetch_repositories, Repository, Developer
from cognee.tasks.github.fetch_contributors import fetch_contributors, Contribution
from cognee.tasks.github.collaboration_network import build_collaboration_network, analyze_collaboration_network
from cognee.tasks.github.pull_requests import fetch_pull_requests, PullRequest, PRComment, DeveloperInteraction
from cognee.tasks.github.developer_personality import analyze_developer_personality
from cognee.tasks.github.readme_analysis import fetch_readme, ReadmeDocument
from cognee.tasks.github.temporal_analysis import analyze_temporal_activities, analyze_developer_trajectory
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.engine import DataPoint

monitoring = get_base_config().monitoring_tool
if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


logger = logging.getLogger(__name__)
update_status_lock = asyncio.Lock()


# Helper function to process data points in batches
async def process_batch(data_points_batch):
    if data_points_batch:
        # Add data points to the graph
        await add_data_points(data_points_batch)
        # Return empty batch
        return []
    return data_points_batch


# Helper function to add a data point to the batch
async def add_to_batch(data_point, data_points_batch, batch_size=50):
    data_points_batch.append(data_point)
    # Yield the data point to the caller
    yield data_point
    # Process the batch if it reaches the batch size
    if len(data_points_batch) >= batch_size:
        processed_batch = await process_batch(data_points_batch)
        # Instead of returning, we modify the input list reference
        data_points_batch.clear()
        data_points_batch.extend(processed_batch)
    # No return statement here


async def fetch_developer_repositories(username: str, api_token: str = None, max_repos: int = 5):
    """Fetch repositories for a GitHub developer."""
    repositories = []
    developers = {}
    data_points_batch = []
    
    # Fetch repositories for the developer
    async for data_point in fetch_repositories(username, api_token=api_token, max_repos=max_repos):
        if isinstance(data_point, Repository):
            repositories.append(data_point)
        elif isinstance(data_point, Developer):
            developers[str(data_point.id)] = data_point

        # Add the data point to the graph batch
        async for _ in add_to_batch(data_point, data_points_batch):
            pass
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)
    
    return repositories, developers


async def fetch_repository_contributors(repositories: List[Repository], api_token: str = None, max_contributors: int = 10):
    """Fetch contributors for each repository."""
    developers = {}
    contributions = []
    repo_contributors = {}  # repo_id -> [contributor_ids]
    data_points_batch = []
    
    # Process each repository for more detailed analysis
    for repo in repositories:
        # Fetch contributors for this repository
        repo_contributors[str(repo.id)] = []
        
        async for data_point in fetch_contributors(repo, api_token=api_token, max_contributors=max_contributors):
            if isinstance(data_point, Developer):
                developers[str(data_point.id)] = data_point
            elif isinstance(data_point, Contribution):
                contributions.append(data_point)
                repo_contributors[str(repo.id)].append(data_point.developer_id)
            
            # Add the data point to the graph batch
            async for _ in add_to_batch(data_point, data_points_batch):
                pass
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)
    
    return developers, contributions, repo_contributors


async def fetch_repository_readmes(repositories: List[Repository], api_token: str = None):
    """Fetch README documents for each repository."""
    readme_docs = []
    data_points_batch = []
    
    for repo in repositories:
        # Fetch README for this repository
        async for data_point in fetch_readme(repo, api_token=api_token):
            readme_docs.append(data_point)
            async for _ in add_to_batch(data_point, data_points_batch):
                pass
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)
    
    return readme_docs


async def summarize_readmes(readme_docs: List[ReadmeDocument]):
    """Generate LLM summaries for README documents."""
    data_points_batch = []
    
    for readme in readme_docs:
        if not readme.content or len(readme.content.strip()) < 10:
            continue
            
        # Generate a concise summary of the README
        prompt = f"""
        Please provide a concise summary of the following repository README:
        
        {readme.content[:4000]}  # Limit content to prevent token overflow
        
        Focus on:
        1. The main purpose of the repository
        2. Key features or capabilities
        3. Technologies used
        
        Keep your response under 200 words.
        """
        
        try:
            summary = await get_llm_response(prompt)
            
            # Create a DataPoint for the summary
            summary_id = f"{readme.id}_summary"
            summary_point = DataPoint(
                id=summary_id,
                name=f"Summary of {readme.repository_name}",
                type="TextSummary",
                content=summary
            )
            
            # Create an Edge connecting the summary to the README
            edge = Edge(
                source_node_id=summary_id,
                target_node_id=readme.id,
                relationship_name="summarizes"
            )
            
            # Add data points to batch
            async for _ in add_to_batch(summary_point, data_points_batch):
                pass
            async for _ in add_to_batch(edge, data_points_batch):
                pass
            
            yield summary_point
            yield edge
            
        except Exception as e:
            logger.error(f"Error generating summary for README {readme.id}: {e}")
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


async def fetch_repository_pull_requests(repositories: List[Repository], api_token: str = None, max_prs: int = 5, max_comments_per_pr: int = 20):
    """Fetch pull requests and comments for each repository."""
    pull_requests = []
    pr_comments = []
    interactions = []
    developers = {}
    data_points_batch = []
    
    for repo in repositories:
        # Fetch pull requests and their comments
        async for data_point in fetch_pull_requests(repo, api_token=api_token, max_prs=max_prs, max_comments_per_pr=max_comments_per_pr):
            if isinstance(data_point, PullRequest):
                pull_requests.append(data_point)
            elif isinstance(data_point, PRComment):
                pr_comments.append(data_point)
            elif isinstance(data_point, DeveloperInteraction):
                interactions.append(data_point)
            elif isinstance(data_point, Developer):
                developers[str(data_point.id)] = data_point
            
            # Add the data point to the graph batch
            async for _ in add_to_batch(data_point, data_points_batch):
                pass
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)
    
    return pull_requests, pr_comments, interactions, developers


async def analyze_pr_sentiment(pr_comments: List[PRComment]):
    """Analyze sentiment of pull request comments using LLM."""
    data_points_batch = []
    
    # Group comments by PR to analyze them together
    pr_comment_map = {}
    for comment in pr_comments:
        if comment.pull_request_id not in pr_comment_map:
            pr_comment_map[comment.pull_request_id] = []
        pr_comment_map[comment.pull_request_id].append(comment)
    
    # Process each PR's comments for sentiment analysis
    for pr_id, comments in pr_comment_map.items():
        if not comments or len(comments) < 2:  # Need at least a couple of comments for meaningful analysis
            continue
            
        # Prepare comments for analysis
        comment_text = "\n\n".join([
            f"Comment by {comment.author_login}: {comment.body[:500]}"  # Limit comment size
            for comment in comments[:5]  # Limit number of comments to analyze
        ])
        
        prompt = f"""
        Please analyze the sentiment and tone of the following pull request discussion:
        
        {comment_text}
        
        Provide a brief assessment of:
        1. Overall sentiment (positive, neutral, negative)
        2. Tone of the discussion (collaborative, critical, contentious, etc.)
        3. Key themes or points of focus
        
        Keep your analysis concise, under 150 words.
        """
        
        try:
            sentiment_analysis = await get_llm_response(prompt)
            
            # Create a DataPoint for the sentiment analysis
            sentiment_id = f"{pr_id}_sentiment"
            sentiment_point = DataPoint(
                id=sentiment_id,
                name=f"Sentiment Analysis for PR #{pr_id}",
                type="SentimentAnalysis",
                content=sentiment_analysis
            )
            
            # Create an Edge connecting the sentiment to the PR
            edge = Edge(
                source_node_id=sentiment_id,
                target_node_id=pr_id,
                relationship_name="analyzes_sentiment_of"
            )
            
            # Add data points to batch
            async for _ in add_to_batch(sentiment_point, data_points_batch):
                pass
            async for _ in add_to_batch(edge, data_points_batch):
                pass
            
            yield sentiment_point
            yield edge
            
        except Exception as e:
            logger.error(f"Error generating sentiment analysis for PR {pr_id}: {e}")
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


async def build_collaboration_graph(repositories, developers, repo_contributors):
    """Build collaboration network graph."""
    data_points_batch = []
    
    # Build collaboration network
    async for data_point in build_collaboration_network(
        repositories, developers, repo_contributors
    ):
        async for _ in add_to_batch(data_point, data_points_batch):
            pass
        yield data_point
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


async def analyze_developer_network(developers, collaborations=None):
    """Analyze developer collaboration network metrics."""
    if collaborations is None:
        collaborations = []
        
    data_points_batch = []
    
    # Analyze collaboration network metrics
    async for data_point in analyze_collaboration_network(
        developers, collaborations
    ):
        async for _ in add_to_batch(data_point, data_points_batch):
            pass
        yield data_point
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


async def analyze_developers_personality(developers, pull_requests, pr_comments, interactions, contributions):
    """Analyze developer personalities based on their GitHub activities."""
    data_points_batch = []
    
    # Create mappings needed for personality analysis
    dev_prs = {dev_id: [] for dev_id in developers}
    dev_comments = {dev_id: [] for dev_id in developers}
    dev_interactions = {dev_id: [] for dev_id in developers}
    dev_contribution_counts = {dev_id: {} for dev_id in developers}
    
    # Populate the mappings
    for pr in pull_requests:
        if pr.creator_id in dev_prs:
            dev_prs[pr.creator_id].append(pr)
    
    for comment in pr_comments:
        if comment.author_id in dev_comments:
            dev_comments[comment.author_id].append(comment)
    
    for interaction in interactions:
        if interaction.developer1_id in dev_interactions:
            dev_interactions[interaction.developer1_id].append(interaction)
        if interaction.developer2_id in dev_interactions:
            dev_interactions[interaction.developer2_id].append(interaction)
    
    for contribution in contributions:
        dev_id = contribution.developer_id
        repo_id = contribution.repository_id
        if dev_id in dev_contribution_counts:
            dev_contribution_counts[dev_id][repo_id] = contribution.contributions_count
    
    # Analyze developer personalities
    for dev_id, developer in developers.items():
        async for data_point in analyze_developer_personality(
            developer,
            dev_prs.get(dev_id, []),
            dev_comments.get(dev_id, []),
            dev_interactions.get(dev_id, []),
            dev_contribution_counts.get(dev_id, {})
        ):
            async for _ in add_to_batch(data_point, data_points_batch):
                pass
            yield data_point
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


async def analyze_temporal_patterns(developers, pull_requests, pr_comments, interactions):
    """Analyze temporal activity patterns of developers."""
    data_points_batch = []
    
    # Create mappings needed for temporal analysis
    dev_prs = {dev_id: [] for dev_id in developers}
    dev_comments = {dev_id: [] for dev_id in developers}
    dev_interactions = {dev_id: [] for dev_id in developers}
    
    # Populate the mappings
    for pr in pull_requests:
        if pr.creator_id in dev_prs:
            dev_prs[pr.creator_id].append(pr)
    
    for comment in pr_comments:
        if comment.author_id in dev_comments:
            dev_comments[comment.author_id].append(comment)
    
    for interaction in interactions:
        if interaction.developer1_id in dev_interactions:
            dev_interactions[interaction.developer1_id].append(interaction)
        if interaction.developer2_id in dev_interactions:
            dev_interactions[interaction.developer2_id].append(interaction)
    
    # Create empty contribution dates mapping for now
    contribution_dates = {dev_id: {} for dev_id in developers}
    
    # Analyze temporal activities for each developer
    for dev_id, developer in developers.items():
        async for data_point in analyze_temporal_activities(
            developer,
            dev_prs.get(dev_id, []),
            dev_comments.get(dev_id, []),
            dev_interactions.get(dev_id, []),
            contribution_dates.get(dev_id, {})
        ):
            async for _ in add_to_batch(data_point, data_points_batch):
                pass
            yield data_point
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


async def generate_developer_summary(developer: Developer, repositories, pull_requests, pr_comments, contributions):
    """Generate an LLM-based summary of the developer's GitHub profile and activities."""
    data_points_batch = []
    
    # Filter repositories, PRs, and comments for this developer
    dev_repos = [repo for repo in repositories if repo.owner == developer.username]
    dev_prs = [pr for pr in pull_requests if pr.creator_id == developer.id]
    dev_comments = [comment for comment in pr_comments if comment.author_id == developer.id]
    dev_contribs = [contrib for contrib in contributions if contrib.developer_id == developer.id]
    
    # Prepare input for the LLM
    repo_info = "\n".join([f"- {repo.name}: {repo.description or 'No description'}" for repo in dev_repos[:5]])
    pr_count = len(dev_prs)
    comment_count = len(dev_comments)
    contrib_repos = len(set(contrib.repository_id for contrib in dev_contribs))
    
    prompt = f"""
    Please provide a professional developer profile summary based on the following GitHub data:
    
    Developer: {developer.name or developer.username}
    GitHub Username: {developer.username}
    Bio: {developer.bio or 'Not provided'}
    
    Notable Repositories ({len(dev_repos)} total):
    {repo_info}
    
    Activity Summary:
    - Created {pr_count} pull requests
    - Made {comment_count} comments on issues/PRs
    - Contributed to {contrib_repos} repositories
    
    Based on this information, provide a concise professional summary of this developer's:
    1. Technical focus areas and expertise
    2. Working style and collaboration patterns
    3. Notable contributions or projects
    
    Keep your response under 200 words and focus on professional insights.
    """
    
    try:
        # Generate a proper UUID for the developer summary using uuid5
        summary_id = uuid5(NAMESPACE_OID, f"{developer.id}_profile_summary")
        
        # Generate the summary using LLM
        try:
            # Use acreate_structured_output instead of acreate
            dev_summary = await get_llm_response(prompt)
        except Exception as llm_error:
            logger.error(f"Error getting LLM response: {llm_error}")
            # Provide a fallback summary if LLM fails
            dev_summary = f"Developer {developer.username} has contributed to {contrib_repos} repositories, created {pr_count} pull requests, and made {comment_count} comments."
        
        # Create a DataPoint for the developer summary with a properly formatted UUID
        display_name = f"Summary: {developer.username}"
        summary_point = DataPoint(
            id=summary_id,
            name=display_name,
            type="DeveloperSummary",
            content=dev_summary,
            username=developer.username,
            developer_id=str(developer.id),
            repository_count=len(dev_repos),
            pr_count=pr_count,
            comment_count=comment_count
        )
        
        # Create an Edge connecting the summary to the developer
        edge = Edge(
            source_node_id=str(summary_id),
            target_node_id=str(developer.id),
            relationship_name="summarizes_profile_of",
            properties={
                "weight": 1.0,
                "name": f"Profile summary for {developer.username}",
                "description": "Summarizes developer's GitHub profile"
            }
        )
        
        # Add data points to batch
        async for _ in add_to_batch(summary_point, data_points_batch):
            pass
        async for _ in add_to_batch(edge, data_points_batch):
            pass
        
        yield summary_point
        yield edge
        
    except Exception as e:
        logger.error(f"Error generating profile summary for developer {developer.id}: {e}")
    
    # Process any remaining data points in the batch
    await process_batch(data_points_batch)


@observe
async def run_github_developer_pipeline(github_username: str, api_token: str = None, run_tasks_list: List[str] = None) -> AsyncGenerator[Any, None]:
    """
    Run the GitHub developer analysis pipeline for a specified GitHub username.
    
    Args:
        github_username: GitHub username to analyze
        api_token: Optional GitHub API token for authentication
        run_tasks_list: Optional list of specific tasks to run (if None, runs all tasks)
    """
    import cognee
    from cognee.low_level import setup
    from cognee.tasks.storage import add_data_points

    await prune_data()
    await prune_system(metadata=True)
    await setup()

    cognee_config = get_cognify_config()
    user = await get_default_user()

    # Define all possible pipeline tasks
    all_task_names = [
        "fetch_repositories",
        "fetch_contributors",
        "fetch_readmes",
        "summarize_readmes",
        "fetch_pull_requests",
        "analyze_pr_sentiment",
        "build_collaboration",
        "analyze_network",
        "analyze_personality",
        "analyze_temporal",
        "generate_developer_summary"
    ]
    
    # Filter tasks to run if list provided
    task_names_to_run = run_tasks_list if run_tasks_list else all_task_names
    
    # Initialize collections to store data
    repositories = []
    developers = {}
    contributions = []
    repo_contributors = {}
    readme_docs = []
    pull_requests = []
    pr_comments = []
    interactions = []
    collaborations = []
    
    # Create the pipeline
    try:
        # Step 1: Fetch repositories
        if "fetch_repositories" in task_names_to_run:
            yield "Starting task: fetch_repositories"
            all_data_points = []
            
            # Create an async generator from the coroutine
            async for data_point in fetch_repositories(github_username, api_token=api_token, max_repos=5):
                if isinstance(data_point, Repository):
                    repositories.append(data_point)
                elif isinstance(data_point, Developer):
                    developers[str(data_point.id)] = data_point
                
                # Collect all data points for adding to the graph
                all_data_points.append(data_point)
                yield f"fetch_repositories: processed {type(data_point).__name__}"
                
                # Save data points in batches of 10
                if len(all_data_points) >= 10:
                    await add_data_points(all_data_points)
                    yield f"Added {len(all_data_points)} data points to graph"
                    all_data_points = []
            
            # Save any remaining data points
            if all_data_points:
                await add_data_points(all_data_points)
                yield f"Added final {len(all_data_points)} data points to graph"
            
            yield f"fetch_repositories: Found {len(repositories)} repositories and {len(developers)} developers"
                
        # Step 2: Fetch contributors
        if "fetch_contributors" in task_names_to_run and repositories:
            yield "Starting task: fetch_contributors"
            contributors_data_points = []
            for repo in repositories:
                async for data_point in fetch_contributors(repo, api_token=api_token, max_contributors=10):
                    if isinstance(data_point, Developer):
                        developers[str(data_point.id)] = data_point
                    elif isinstance(data_point, Contribution):
                        contributions.append(data_point)
                        repo_id = str(repo.id)
                        if repo_id not in repo_contributors:
                            repo_contributors[repo_id] = []
                        repo_contributors[repo_id].append(data_point.developer_id)
                    
                    contributors_data_points.append(data_point)
                    yield f"fetch_contributors: processed {type(data_point).__name__} for repo {repo.name}"
                    
                    # Save data points in batches of 10
                    if len(contributors_data_points) >= 10:
                        await add_data_points(contributors_data_points)
                        yield f"Added {len(contributors_data_points)} contributor data points to graph"
                        contributors_data_points = []
            
            # Save any remaining data points
            if contributors_data_points:
                await add_data_points(contributors_data_points)
                yield f"Added final {len(contributors_data_points)} contributor data points to graph"

        # Step 3: Fetch README documents
        if "fetch_readmes" in task_names_to_run and repositories:
            yield "Starting task: fetch_readmes"
            readme_data_points = []
            
            for repo in repositories:
                async for data_point in fetch_readme(repo, api_token=api_token):
                    if isinstance(data_point, ReadmeDocument):
                        readme_docs.append(data_point)
                    
                    readme_data_points.append(data_point)
                    yield f"fetch_readmes: processed README for repo {repo.name}"
                    
                    # Save data points in batches
                    if len(readme_data_points) >= 10:
                        await add_data_points(readme_data_points)
                        yield f"Added {len(readme_data_points)} README data points to graph"
                        readme_data_points = []
            
            # Save any remaining data points
            if readme_data_points:
                await add_data_points(readme_data_points)
                yield f"Added final {len(readme_data_points)} README data points to graph"
        
        # Step 4: Summarize README documents with LLM
        if "summarize_readmes" in task_names_to_run and readme_docs:
            yield "Starting task: summarize_readmes"
            summary_data_points = []
            
            async for data_point in summarize_readmes(readme_docs):
                summary_data_points.append(data_point)
                yield f"summarize_readmes: processed README summary"
                
                # Save data points in batches
                if len(summary_data_points) >= 5:
                    await add_data_points(summary_data_points)
                    yield f"Added {len(summary_data_points)} README summary data points to graph"
                    summary_data_points = []
            
            # Save any remaining data points
            if summary_data_points:
                await add_data_points(summary_data_points)
                yield f"Added final {len(summary_data_points)} README summary data points to graph"

        # Step 5: Fetch pull requests and comments
        if "fetch_pull_requests" in task_names_to_run and repositories:
            yield "Starting task: fetch_pull_requests"
            pr_data_points = []
            
            for repo in repositories:
                async for data_point in fetch_pull_requests(repo, api_token=api_token, max_prs=5, max_comments_per_pr=20):
                    if isinstance(data_point, PullRequest):
                        pull_requests.append(data_point)
                    elif isinstance(data_point, PRComment):
                        pr_comments.append(data_point)
                    elif isinstance(data_point, DeveloperInteraction):
                        interactions.append(data_point)
                    elif isinstance(data_point, Developer) and str(data_point.id) not in developers:
                        developers[str(data_point.id)] = data_point
                    
                    pr_data_points.append(data_point)
                    yield f"fetch_pull_requests: processed {type(data_point).__name__} for repo {repo.name}"
                    
                    # Save data points in batches
                    if len(pr_data_points) >= 20:
                        await add_data_points(pr_data_points)
                        yield f"Added {len(pr_data_points)} PR data points to graph"
                        pr_data_points = []
            
            # Save any remaining data points
            if pr_data_points:
                await add_data_points(pr_data_points)
                yield f"Added final {len(pr_data_points)} PR data points to graph"
            
            yield f"fetch_pull_requests: Found {len(pull_requests)} PRs, {len(pr_comments)} comments, and {len(interactions)} interactions"

        # Step 6: Analyze PR sentiment with LLM
        if "analyze_pr_sentiment" in task_names_to_run and pull_requests:
            yield "Starting task: analyze_pr_sentiment"
            sentiment_data_points = []
            
            async for data_point in analyze_pr_sentiment(pull_requests, pr_comments):
                sentiment_data_points.append(data_point)
                yield f"analyze_pr_sentiment: processed sentiment analysis"
                
                # Save data points in batches
                if len(sentiment_data_points) >= 5:
                    await add_data_points(sentiment_data_points)
                    yield f"Added {len(sentiment_data_points)} sentiment analysis data points to graph"
                    sentiment_data_points = []
            
            # Save any remaining data points
            if sentiment_data_points:
                await add_data_points(sentiment_data_points)
                yield f"Added final {len(sentiment_data_points)} sentiment analysis data points to graph"

        # Step 7: Build collaboration network
        if "build_collaboration" in task_names_to_run and repositories and developers:
            yield "Starting task: build_collaboration"
            collab_data_points = []
            
            async for data_point in build_collaboration_network(repositories, developers, repo_contributors):
                if hasattr(data_point, "developer1_id") and hasattr(data_point, "developer2_id"):
                    collaborations.append(data_point)
                
                collab_data_points.append(data_point)
                yield f"build_collaboration: processed collaboration data"
                
                # Save data points in batches
                if len(collab_data_points) >= 10:
                    await add_data_points(collab_data_points)
                    yield f"Added {len(collab_data_points)} collaboration data points to graph"
                    collab_data_points = []
            
            # Save any remaining data points
            if collab_data_points:
                await add_data_points(collab_data_points)
                yield f"Added final {len(collab_data_points)} collaboration data points to graph"
            
            yield f"build_collaboration: Created {len(collaborations)} collaboration relationships"

        # Step 8: Analyze network metrics
        if "analyze_network" in task_names_to_run and developers:
            yield "Starting task: analyze_network"
            network_data_points = []
            
            async for data_point in analyze_collaboration_network(developers, collaborations):
                network_data_points.append(data_point)
                yield f"analyze_network: processed network metrics"
                
                # Save data points in batches
                if len(network_data_points) >= 10:
                    await add_data_points(network_data_points)
                    yield f"Added {len(network_data_points)} network metrics data points to graph"
                    network_data_points = []
            
            # Save any remaining data points
            if network_data_points:
                await add_data_points(network_data_points)
                yield f"Added final {len(network_data_points)} network metrics data points to graph"

        # Step 9: Analyze developer personalities
        if "analyze_personality" in task_names_to_run and developers:
            yield "Starting task: analyze_personality"
            personality_data_points = []
            
            async for data_point in analyze_developers_personality(developers, pull_requests, pr_comments, interactions, contributions):
                personality_data_points.append(data_point)
                yield f"analyze_personality: processed personality data"
                
                # Save data points in batches
                if len(personality_data_points) >= 5:
                    await add_data_points(personality_data_points)
                    yield f"Added {len(personality_data_points)} personality data points to graph"
                    personality_data_points = []
            
            # Save any remaining data points
            if personality_data_points:
                await add_data_points(personality_data_points)
                yield f"Added final {len(personality_data_points)} personality data points to graph"

        # Step 10: Analyze temporal patterns
        if "analyze_temporal" in task_names_to_run and developers:
            yield "Starting task: analyze_temporal"
            temporal_data_points = []
            
            async for data_point in analyze_temporal_patterns(developers, pull_requests, pr_comments, interactions):
                temporal_data_points.append(data_point)
                yield f"analyze_temporal: processed temporal data"
                
                # Save data points in batches
                if len(temporal_data_points) >= 5:
                    await add_data_points(temporal_data_points)
                    yield f"Added {len(temporal_data_points)} temporal data points to graph"
                    temporal_data_points = []
            
            # Save any remaining data points
            if temporal_data_points:
                await add_data_points(temporal_data_points)
                yield f"Added final {len(temporal_data_points)} temporal data points to graph"

        # Step 11: Generate developer summaries
        if "generate_developer_summary" in task_names_to_run and developers:
            yield "Starting task: generate_developer_summary"
            summary_data_points = []
            
            for dev_id, developer in developers.items():
                async for data_point in generate_developer_summary(developer, repositories, pull_requests, pr_comments, contributions):
                    summary_data_points.append(data_point)
                    yield f"generate_developer_summary: processed summary for {developer.username}"
                    
                    # Save data points in batches
                    if len(summary_data_points) >= 2:
                        await add_data_points(summary_data_points)
                        yield f"Added {len(summary_data_points)} developer summary data points to graph"
                        summary_data_points = []
            
            # Save any remaining data points
            if summary_data_points:
                await add_data_points(summary_data_points)
                yield f"Added final {len(summary_data_points)} developer summary data points to graph"
    
    except Exception as e:
        logger.error(f"Error in GitHub pipeline: {e}")
        yield f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    async def main():
        # Replace with an actual GitHub username and optionally an API token
        github_username = "octocat"
        api_token = None  # Optional: "ghp_yourtokenhere"
        
        # Run specific tasks only
        tasks_to_run = [
            "fetch_repositories",
            "fetch_contributors",
            "fetch_readmes",
            "summarize_readmes"
        ]
        
        async for data_points in run_github_developer_pipeline(github_username, api_token, tasks_to_run):
            print(data_points)

        # Visualize the graph
        await visualize_graph()

    asyncio.run(main()) 