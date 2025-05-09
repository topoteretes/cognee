import json
import asyncio
import cognee
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_dev_profile import GitHubDevProfile


def get_github_profile_data(
    username, token=None, days=30, prs_limit=5, commits_per_pr=3, issues_limit=5, max_comments=3
):
    """Fetches comprehensive GitHub profile data including user info, commits from PRs, and comments."""
    token = token or ""
    profile = GitHubDevProfile(username, token)

    if not profile.user:
        return None

    commits_result = profile.get_user_commits(
        days=days, prs_limit=prs_limit, commits_per_pr=commits_per_pr, include_files=True
    )
    comments = profile.get_issue_comments(
        days=days, issues_limit=issues_limit, max_comments=max_comments, include_issue_details=True
    )

    return {
        "user": profile.get_user_info(),
        "commits": commits_result["commits"] if commits_result else [],
        "comments": comments or [],
    }


def get_github_file_changes(
    username, token=None, days=30, prs_limit=5, commits_per_pr=3, skip_no_diff=True
):
    """Fetches a flat list of file changes from PRs with associated commit information for a GitHub user."""
    token = token or ""
    profile = GitHubDevProfile(username, token)

    if not profile.user:
        return None

    file_changes = profile.get_user_file_changes(
        days=days, prs_limit=prs_limit, commits_per_pr=commits_per_pr, skip_no_diff=skip_no_diff
    )

    return {"user": profile.get_user_info(), "file_changes": file_changes or []}


def get_github_data_for_cognee(
    username,
    token=None,
    days=30,
    prs_limit=5,
    commits_per_pr=3,
    issues_limit=5,
    max_comments=3,
    skip_no_diff=True,
):
    """Fetches enriched GitHub data for a user with PR file changes and comments combined with user data."""
    token = token or ""
    profile = GitHubDevProfile(username, token)

    if not profile.user:
        return None

    user_info = profile.get_user_info()

    file_changes = profile.get_user_file_changes(
        days=days, prs_limit=prs_limit, commits_per_pr=commits_per_pr, skip_no_diff=skip_no_diff
    )

    enriched_file_changes = []
    if file_changes:
        enriched_file_changes = [item | user_info for item in file_changes]

    comments = profile.get_issue_comments(
        days=days, issues_limit=issues_limit, max_comments=max_comments, include_issue_details=True
    )

    enriched_comments = []
    if comments:
        enriched_comments = [comment | user_info for comment in comments]

    return {"user": user_info, "file_changes": enriched_file_changes, "comments": enriched_comments}


async def cognify_github_profile(username, token=None):
    """Ingests GitHub data into Cognee with soft and technical node sets."""
    github_data = get_github_data_for_cognee(username=username, token=token)
    if not github_data:
        return False

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(json.dumps(github_data["user"], default=str), node_set=["soft", "technical"])

    for comment in github_data["comments"]:
        await cognee.add("Comment: " + json.dumps(comment, default=str), node_set=["soft"])

    for file_change in github_data["file_changes"]:
        await cognee.add(
            "File Change: " + json.dumps(file_change, default=str), node_set=["technical"]
        )

    await cognee.cognify()
    return True


async def main(username):
    """Main function for testing Cognee ingest."""
    import os
    import dotenv
    from cognee.api.v1.visualize.visualize import visualize_graph

    dotenv.load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    await cognify_github_profile(username, token)

    # success = await cognify_github_profile(username, token)

    # if success:
    #     visualization_path = os.path.join(os.path.dirname(__file__), "./.artifacts/github_graph.html")
    #     await visualize_graph(visualization_path)


if __name__ == "__main__":
    import os
    import dotenv

    dotenv.load_dotenv()

    username = ""
    asyncio.run(main(username))
    # token = os.getenv("GITHUB_TOKEN")
    # github_data = get_github_data_for_cognee(username=username, token=token)
    # print(json.dumps(github_data, indent=2, default=str))
