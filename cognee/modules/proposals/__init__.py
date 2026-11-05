"""Governed-model proposals: the "LLM proposes, human ratifies" loop as one mechanism.

``generate`` drafts ``OntologyProposal`` nodes (mapping, definition_conflict,
ontology_extension) from the graph; ``apply`` ratifies or rejects one, writing the
change with ``ontology_valid=True`` and a ``ratified_by`` stamp. The improve stage
``ontology_proposals`` runs the drafting; ``/api/v1/proposals/ontology`` exposes the
review.
"""

from cognee.modules.engine.models.OntologyProposal import (
    PROPOSAL_KIND_DEFINITION_CONFLICT,
    PROPOSAL_KIND_MAPPING,
    PROPOSAL_KIND_ONTOLOGY_EXTENSION,
    PROPOSAL_KINDS,
    PROPOSAL_STATUS_APPLIED,
    PROPOSAL_STATUS_PROPOSED,
    PROPOSAL_STATUS_REJECTED,
    OntologyProposal,
)

from .apply import ProposalNotApplicableError, apply_proposal, reject_proposal
from .generate import generate_ontology_proposals
from .service import (
    get_ontology_proposal,
    list_ontology_proposals,
    ratify_ontology_proposal,
)
from .store import find_proposal, load_proposals, proposal_id_for, save_proposals

__all__ = [
    "PROPOSAL_KINDS",
    "PROPOSAL_KIND_DEFINITION_CONFLICT",
    "PROPOSAL_KIND_MAPPING",
    "PROPOSAL_KIND_ONTOLOGY_EXTENSION",
    "PROPOSAL_STATUS_APPLIED",
    "PROPOSAL_STATUS_PROPOSED",
    "PROPOSAL_STATUS_REJECTED",
    "OntologyProposal",
    "ProposalNotApplicableError",
    "apply_proposal",
    "find_proposal",
    "generate_ontology_proposals",
    "get_ontology_proposal",
    "list_ontology_proposals",
    "load_proposals",
    "proposal_id_for",
    "ratify_ontology_proposal",
    "reject_proposal",
    "save_proposals",
]
