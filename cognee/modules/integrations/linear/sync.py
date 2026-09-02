"""Sync a Linear workspace's issues into cognee memory as text.

The textual mirror of the GitHub adapter's repository sync: where GitHub
feeds clone URLs to the code-graph pipeline, Linear feeds a stable
plain-text rendering of each issue to the ordinary ``remember()`` path, so
issue history becomes searchable knowledge the agent (and any recall) can
draw on. ``format_issue`` keeps that rendering deterministic — the same
issue state always produces the same text, so re-syncs don't churn the
graph with cosmetic differences.

One dataset per workspace (``linear_<url_key>``), not per team or issue —
same reasoning as GitHub's one-dataset-per-installation: per-item datasets
would mean one isolated database per item under backend access control.

Per-issue webhook syncs run with ``self_improvement=False``: ``improve()``
is a whole-graph enrichment pass with LLM cost attached, far too heavy to
fire on every issue edit — enrichment stays a human/scheduled decision, the
webhook path just has to be cheap enough to run on every delivery.
"""

import logging
import re
from typing import Any

from cognee.modules.integrations.linear.client import graphql
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

LINEAR_DATASET_PREFIX = "linear"

_RECENT_ISSUES_QUERY = """
query RecentIssues($limit: Int!) {
  issues(first: $limit, orderBy: updatedAt) {
    nodes { id identifier title description url state { name } }
  }
}
"""


def dataset_name_for_org(url_key: str) -> str:
    """The one dataset a workspace's issues land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", url_key).strip("_").lower()
    return f"{LINEAR_DATASET_PREFIX}_{slug or 'workspace'}"


def format_issue(issue: dict[str, Any]) -> str:
    """A stable plain-text rendering of one issue.

    Identifier, title, URL, state, description — nothing volatile (no
    updated-at timestamps, no computed counts), so an unchanged issue always
    renders to byte-identical text.
    """
    identifier = issue.get("identifier") or issue.get("id") or "unknown"
    state_name = (issue.get("state") or {}).get("name") or "Unknown"

    lines = [
        f"Linear issue {identifier}: {issue.get('title') or ''}".rstrip(),
        f"URL: {issue.get('url') or ''}",
        f"State: {state_name}",
    ]
    description = issue.get("description")
    if description:
        lines.append(f"Description: {description}")
    return "\n".join(lines)


def _dataset_name(credential: IntegrationCredential) -> str:
    url_key = (credential.provider_metadata or {}).get("organization_url_key") or str(
        credential.provider_account_id
    )
    return dataset_name_for_org(url_key)


async def sync_issue(credential: IntegrationCredential, issue: dict[str, Any]) -> None:
    """Remember one issue's current text (webhook-driven create/update path).

    Runs the ingestion to completion — callers are already off the request
    path (webhook handling runs detached), so there is nothing to hand off
    to.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.api.v1.remember.remember import remember as cognee_remember
    from cognee.modules.users.methods import get_user

    owner = await get_user(credential.user_id)
    result = await cognee_remember(
        format_issue(issue),
        dataset_name=_dataset_name(credential),
        user=owner,
        # Per-issue webhook syncs must stay cheap; improve() is a
        # human/scheduled decision (see module docstring).
        self_improvement=False,
    )
    if getattr(result, "status", None) == "errored":
        logger.warning(
            "Linear issue sync for organization %s finished with errors: %s",
            credential.provider_account_id,
            getattr(result, "error", None),
        )


async def sync_recent_issues(credential: IntegrationCredential, limit: int = 50) -> None:
    """Remember the workspace's most recently updated issues, one batch.

    The post-install seed: gives the agent something to recall from on day
    one, since webhooks only cover changes from now on. One ``remember()``
    call for the whole batch, not one per issue — a single pipeline run over
    the list is far cheaper than fifty.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.api.v1.remember.remember import remember as cognee_remember
    from cognee.modules.users.methods import get_user

    # Imported here, not at module top: the adapter imports this module to
    # wire on_installed, so a top-level import back into it would be
    # circular.
    from cognee.modules.integrations.linear.adapter import access_token_for

    data = await graphql(access_token_for(credential), _RECENT_ISSUES_QUERY, {"limit": limit})
    issues = ((data.get("issues") or {}).get("nodes")) or []
    if not issues:
        logger.info("Linear organization %s has no issues to sync", credential.provider_account_id)
        return

    owner = await get_user(credential.user_id)
    dataset_name = _dataset_name(credential)

    logger.info(
        "Syncing %d Linear issues for organization %s into dataset %s",
        len(issues),
        credential.provider_account_id,
        dataset_name,
    )
    result = await cognee_remember(
        [format_issue(issue) for issue in issues if issue],
        dataset_name=dataset_name,
        user=owner,
        # Same stance as sync_issue: the initial seed should not silently
        # spend an improve() pass either.
        self_improvement=False,
    )
    if getattr(result, "status", None) == "errored":
        logger.warning(
            "Linear initial sync for organization %s finished with errors: %s",
            credential.provider_account_id,
            getattr(result, "error", None),
        )
