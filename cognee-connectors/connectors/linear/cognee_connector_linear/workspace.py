"""What a connected Linear workspace is to cognee, without a host.

Split out of cognee.modules.integrations.linear.sync (cognee dev, 2026-10-26): the parts
of that module that import nothing from cognee — the workspace's dataset name, the stored
team selection, the teams query and the webhook event helpers — so that ``connector.py``
can use them with cognee absent. ``cognee_host/sync.py`` keeps the sync itself and
re-exports these names for the adapter, the event handler and the resume worker.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

LINEAR_DATASET_PREFIX = "linear"
PROVIDER = "linear"

# provider_metadata keys. Selection is three-state like Drive's and Gmail's:
# absent/None means every team the app was granted, [] none, a list those teams.
SELECTION_KEY = "selected_team_ids"

TEAMS_PAGE_SIZE = 100

TEAMS_QUERY = """
query LinearTeams($first: Int!, $after: String) {
  teams(first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { id key name description visibility }
  }
}
"""

_EVENT_TYPES = ("Issue", "Comment", "Project")
_EVENT_ACTIONS = ("create", "update", "remove")


def dataset_name_for_org(url_key: str) -> str:
    """The one dataset a workspace's issues land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", url_key).strip("_").lower()
    return f"{LINEAR_DATASET_PREFIX}_{slug or 'workspace'}"


def dataset_name_for_credential(credential: Any) -> str:
    """``linear_<url key>`` from the connection's stored metadata, the account id failing that."""
    url_key = (credential.provider_metadata or {}).get("organization_url_key") or str(
        credential.provider_account_id
    )
    return dataset_name_for_org(url_key)


def selected_team_ids(credential: Any) -> list[str] | None:
    """The stored team selection: ``None`` for every granted team, ``[]`` for none, else those.

    Anything but a list of non-empty strings is refused: the selection is written by the
    generic resource picker and read by every sync, so a malformed one must fail here, once,
    rather than sync the wrong teams.
    """
    selected = (credential.provider_metadata or {}).get(SELECTION_KEY)
    if selected is not None and (
        not isinstance(selected, list) or not all(isinstance(t, str) and t for t in selected)
    ):
        raise ValueError("Linear selection must be a list of team IDs or null")
    return selected


async def list_teams_with(
    execute: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """The teams the app token can access, paged through ``execute(variables)``.

    ``execute`` runs :data:`TEAMS_QUERY` with whatever token its caller manages — cognee's
    sync takes a fresh one per request, the connector hands over the stored one — so the
    paging and the de-duplication live here once.
    """
    teams: dict[str, dict[str, Any]] = {}
    after = None
    while True:
        data = await execute({"first": TEAMS_PAGE_SIZE, "after": after})
        connection = data["teams"]
        for node in connection["nodes"]:
            teams.setdefault(node["id"], node)
        if not connection["pageInfo"]["hasNextPage"]:
            return list(teams.values())
        after = connection["pageInfo"]["endCursor"]


def describe_team(team: dict[str, Any]) -> dict[str, Any]:
    """A team as a picker resource: its id, name, description and display attributes."""
    return {
        "id": str(team["id"]),
        "name": str(team.get("name") or team["id"]),
        "description": team.get("description"),
        "attributes": {
            "key": team.get("key"),
            "private": team.get("visibility") != "public",
        },
    }


def team_ids_from_event(payload: dict[str, Any]) -> list[str]:
    """The teams a webhook delivery names, whichever entity it is about."""
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        return []
    event_type = payload.get("type")
    if event_type == "Issue":
        candidates = [data.get("teamId")]
    elif event_type == "Comment":
        issue = data.get("issue")
        candidates = [issue.get("teamId") if isinstance(issue, dict) else None, data.get("teamId")]
    elif event_type == "Project":
        team_ids = data.get("teamIds")
        candidates = list(team_ids) if isinstance(team_ids, list) else []
    else:
        return []
    return list(dict.fromkeys(c for c in candidates if isinstance(c, str) and c))


def is_team_event(payload: dict[str, Any]) -> bool:
    return payload.get("type") in _EVENT_TYPES and payload.get("action") in _EVENT_ACTIONS
