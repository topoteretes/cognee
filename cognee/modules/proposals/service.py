"""Dataset-scoped entry points for reviewing ontology proposals (SDK and HTTP share them).

Each function resolves the dataset's database context itself, so callers pass the
authorized ``Dataset`` row and the acting user and nothing else.
"""

from __future__ import annotations

from typing import Any

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.engine.models import OntologyProposal

from .apply import apply_proposal, reject_proposal
from .store import find_proposal, load_proposals


def _owner_id(dataset: Any, user: Any):
    owner_id = getattr(dataset, "owner_id", None) or getattr(user, "id", None)
    if owner_id is None:
        raise ValueError("Ontology proposals require a dataset owner or user.")
    return owner_id


async def list_ontology_proposals(
    *, dataset: Any, user: Any, kind: str | None = None, status: str | None = None
) -> list[OntologyProposal]:
    async with set_database_global_context_variables(dataset.id, _owner_id(dataset, user)):
        return await load_proposals(dataset.id, kind=kind, status=status)


async def get_ontology_proposal(
    proposal_id: str, *, dataset: Any, user: Any
) -> OntologyProposal | None:
    async with set_database_global_context_variables(dataset.id, _owner_id(dataset, user)):
        return await find_proposal(proposal_id, dataset.id)


async def ratify_ontology_proposal(
    proposal_id: str,
    *,
    dataset: Any,
    user: Any,
    decision: str,
    resolution: str = "",
    ratified_by: str | None = None,
) -> OntologyProposal | None:
    """Apply (``decision="apply"``) or reject a proposal; ``None`` when it does not exist."""
    if decision not in ("apply", "reject"):
        raise ValueError(f"decision must be 'apply' or 'reject', got {decision!r}")
    who = ratified_by or getattr(user, "email", None) or str(getattr(user, "id", "unknown"))
    async with set_database_global_context_variables(dataset.id, _owner_id(dataset, user)):
        proposal = await find_proposal(proposal_id, dataset.id)
        if proposal is None:
            return None
        if decision == "apply":
            return await apply_proposal(
                proposal, ratified_by=who, user=user, dataset=dataset, resolution=resolution
            )
        return await reject_proposal(
            proposal, ratified_by=who, user=user, dataset=dataset, resolution=resolution
        )
