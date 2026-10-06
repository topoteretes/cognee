"""Draft, apply and reject governed-model proposals against a fake graph engine."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from rdflib import OWL, RDF, Graph, Namespace

from cognee.modules.engine.models import EntityType
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.proposals import (
    PROPOSAL_KIND_DEFINITION_CONFLICT,
    PROPOSAL_KIND_MAPPING,
    PROPOSAL_KIND_ONTOLOGY_EXTENSION,
    PROPOSAL_STATUS_APPLIED,
    PROPOSAL_STATUS_REJECTED,
    ProposalNotApplicableError,
    apply_proposal,
    generate_ontology_proposals,
    reject_proposal,
)
from cognee.modules.proposals import store as store_module
from cognee.modules.proposals.generate import propose_definition_conflicts

NS = Namespace("http://example.org/crm#")
DATASET_ID = uuid4()


class _FakeGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes  # (id, props)
        self.edges = edges  # (src, tgt, rel, props)
        self.updated = {}

    async def get_filtered_graph_data(self, attribute_filters):
        wanted = set(attribute_filters[0]["type"])
        nodes = [(i, p) for i, p in self.nodes if p.get("type") in wanted]
        ids = {i for i, _ in nodes}
        edges = [e for e in self.edges if e[0] in ids and e[1] in ids]
        return nodes, edges

    async def get_graph_data(self):
        return self.nodes, self.edges

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        edges = [
            e
            for e in self.edges
            if (e[0] in node_ids or e[1] in node_ids) and (not edge_types or e[2] in edge_types)
        ]
        return [], edges

    async def update_node(self, node_id, values):
        for i, props in self.nodes:
            if i == node_id:
                props.update(values)
                self.updated[node_id] = values
                return True
        return False


def _resolver():
    graph = Graph()
    for name in ("Customer", "Company", "Subscription"):
        graph.add((NS[name], RDF.type, OWL.Class))
    resolver = RDFLibOntologyResolver(ontology_file=None)
    resolver.graph = graph
    resolver.build_lookup()
    return resolver


def _fixture_graph():
    customer_type = str(EntityType.id_for("Customer"))
    ticket_type = str(EntityType.id_for("support ticket"))
    nodes = [
        ("t-crm", {"type": "SchemaTable", "name": "crm.customers", "columns": "[]"}),
        (
            "t-arch",
            {
                "type": "SchemaTable",
                "name": "archived_customers",
                "columns": '[{"name": "cust_id"}]',
                "description": "old customers",
            },
        ),
        ("t-log", {"type": "SchemaTable", "name": "zz_audit_log", "columns": "[]"}),
        (customer_type, {"type": "EntityType", "name": "customer", "ontology_valid": True}),
        (ticket_type, {"type": "EntityType", "name": "support ticket"}),
        ("e1", {"type": "Entity", "name": "ticket 1"}),
        ("e2", {"type": "Entity", "name": "ticket 2"}),
        ("e3", {"type": "Entity", "name": "ticket 3"}),
        ("e-a", {"type": "Entity", "name": "acme"}),
        ("e-b", {"type": "Entity", "name": "acme corp"}),
    ]
    edges = [
        ("t-crm", customer_type, "realizes", {"authoritative": True}),
        ("e1", ticket_type, "is_a", {}),
        ("e2", ticket_type, "is_a", {}),
        ("e3", ticket_type, "is_a", {}),
        (
            "e-a",
            "e-b",
            "contradicts",
            {
                "first_fact": "acme is based in Berlin",
                "second_fact": "acme corp is based in Paris",
                "reason": "different headquarters",
                "confidence": 0.9,
            },
        ),
    ]
    return _FakeGraph(nodes, edges), ticket_type


@pytest.mark.asyncio
async def test_generate_drafts_one_proposal_per_kind_and_dedupes():
    graph, ticket_type = _fixture_graph()

    proposals = await generate_ontology_proposals(
        DATASET_ID, resolver=_resolver(), existing_ids=set(), use_llm=False, graph_engine=graph
    )
    by_kind = {proposal.kind: proposal for proposal in proposals}

    assert set(by_kind) == {
        PROPOSAL_KIND_MAPPING,
        PROPOSAL_KIND_DEFINITION_CONFLICT,
        PROPOSAL_KIND_ONTOLOGY_EXTENSION,
    }
    mapping = by_kind[PROPOSAL_KIND_MAPPING]
    assert mapping.subject_id == "t-arch"  # crm.customers is mapped; zz_audit_log is unknown
    assert mapping.concept_name == "Customer"
    assert mapping.proposed_by == "heuristic"
    conflict = by_kind[PROPOSAL_KIND_DEFINITION_CONFLICT]
    assert (conflict.subject_id, conflict.counterpart_id) == ("e-a", "e-b")
    assert conflict.evidence == ["acme is based in Berlin", "acme corp is based in Paris"]
    assert conflict.confidence == 0.9
    extension = by_kind[PROPOSAL_KIND_ONTOLOGY_EXTENSION]
    assert extension.subject_id == ticket_type
    assert extension.occurrences == 3
    assert all(str(DATASET_ID) in proposal.dataset_scope for proposal in proposals)

    # Deterministic ids: a second pass with the first pass's ids drafts nothing new.
    again = await generate_ontology_proposals(
        DATASET_ID,
        resolver=_resolver(),
        existing_ids={proposal.proposal_id for proposal in proposals},
        use_llm=False,
        graph_engine=graph,
    )
    assert again == []


@pytest.mark.asyncio
async def test_extension_threshold_and_resolved_conflicts_are_respected():
    graph, _ = _fixture_graph()
    graph.edges[-1][3]["resolution"] = "Berlin stands"

    proposals = await generate_ontology_proposals(
        DATASET_ID,
        resolver=_resolver(),
        existing_ids=set(),
        use_llm=False,
        min_occurrences=4,
        graph_engine=graph,
    )
    assert {proposal.kind for proposal in proposals} == {PROPOSAL_KIND_MAPPING}


def _saved(monkeypatch):
    calls = []

    async def fake_save(proposals, *, user, dataset, custom_edges=None):
        calls.append((list(proposals), custom_edges))

    monkeypatch.setattr("cognee.modules.proposals.apply.save_proposals", fake_save)
    return calls


@pytest.mark.asyncio
async def test_apply_mapping_writes_a_ratified_realizes_edge(monkeypatch):
    graph, _ = _fixture_graph()
    calls = _saved(monkeypatch)
    proposals = await generate_ontology_proposals(
        DATASET_ID, resolver=_resolver(), existing_ids=set(), use_llm=False, graph_engine=graph
    )
    mapping = next(p for p in proposals if p.kind == PROPOSAL_KIND_MAPPING)

    applied = await apply_proposal(
        mapping,
        ratified_by="steward@example.com",
        user=SimpleNamespace(id=uuid4()),
        dataset=SimpleNamespace(id=DATASET_ID),
        graph_engine=graph,
    )

    assert applied.status == PROPOSAL_STATUS_APPLIED
    assert applied.ratified_by == "steward@example.com"
    ((nodes, edges),) = calls
    concept = next(node for node in nodes if isinstance(node, EntityType))
    assert concept.id == EntityType.id_for("Customer") and concept.ontology_valid is True
    ((source, target, relationship, props),) = edges
    assert (source, target, relationship) == ("t-arch", concept.id, "realizes")
    assert props["target_node_id"] == str(concept.id)
    assert props["ontology_valid"] is True
    assert props["ratified_by"] == "steward@example.com"
    assert props["authoritative"] is False

    with pytest.raises(ProposalNotApplicableError):
        await apply_proposal(
            applied,
            ratified_by="x",
            user=SimpleNamespace(id=uuid4()),
            dataset=SimpleNamespace(id=DATASET_ID),
            graph_engine=graph,
        )


@pytest.mark.asyncio
async def test_apply_conflict_merges_resolution_into_the_existing_edge(monkeypatch):
    graph, _ = _fixture_graph()
    calls = _saved(monkeypatch)
    snapshot_nodes = {i: {"id": i, **p} for i, p in graph.nodes}
    from cognee.modules.proposals.generate import GraphSnapshot

    conflict = propose_definition_conflicts(
        GraphSnapshot(nodes=snapshot_nodes, edges=graph.edges), DATASET_ID
    )[0]

    await apply_proposal(
        conflict,
        ratified_by="steward",
        user=SimpleNamespace(id=uuid4()),
        dataset=SimpleNamespace(id=DATASET_ID),
        resolution="Berlin stands; Paris is the sales office",
        graph_engine=graph,
    )

    ((_, edges),) = calls
    ((_, _, relationship, props),) = edges
    assert relationship == "contradicts"
    assert props["first_fact"] == "acme is based in Berlin"  # merged, not clobbered
    assert props["resolution"] == "Berlin stands; Paris is the sales office"
    assert props["ratified_by"] == "steward"


@pytest.mark.asyncio
async def test_apply_extension_marks_the_entity_type_valid_and_reject_only_records(monkeypatch):
    graph, ticket_type = _fixture_graph()
    calls = _saved(monkeypatch)
    proposals = await generate_ontology_proposals(
        DATASET_ID, resolver=_resolver(), existing_ids=set(), use_llm=False, graph_engine=graph
    )
    extension = next(p for p in proposals if p.kind == PROPOSAL_KIND_ONTOLOGY_EXTENSION)
    conflict = next(p for p in proposals if p.kind == PROPOSAL_KIND_DEFINITION_CONFLICT)
    user, dataset = SimpleNamespace(id=uuid4()), SimpleNamespace(id=DATASET_ID)

    await apply_proposal(
        extension, ratified_by="steward", user=user, dataset=dataset, graph_engine=graph
    )
    assert graph.updated[ticket_type]["ontology_valid"] is True
    assert graph.updated[ticket_type]["ratified_by"] == "steward"

    rejected = await reject_proposal(conflict, ratified_by="steward", user=user, dataset=dataset)
    assert rejected.status == PROPOSAL_STATUS_REJECTED
    # Rejecting writes the proposal only: no concept nodes, no edges.
    nodes, edges = calls[-1]
    assert [type(node).__name__ for node in nodes] == ["OntologyProposal"] and edges is None


def test_coerce_proposal_reads_raw_graph_tuples():
    proposal_id = store_module.proposal_id_for("mapping", DATASET_ID, "t", "Customer")
    raw = (
        str(uuid4()),
        {
            "type": "OntologyProposal",
            "proposal_id": proposal_id,
            "kind": "mapping",
            "dataset_scope": [str(DATASET_ID)],
            "subject_name": "t",
        },
    )
    proposal = store_module.coerce_proposal(raw)
    assert proposal is not None and proposal.proposal_id == proposal_id
    assert store_module.coerce_proposal(("n", {"type": "Entity", "name": "x"})) is None
