"""Read-time ontology grounding: query terms -> ontology nodes -> seeds + context."""

from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology.query_grounding import (
    QueryGrounding,
    extract_candidate_terms,
    ground_query,
    ground_query_with_configured_ontology,
)
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

NS = Namespace("http://example.org/fin#")


def _finance_resolver() -> RDFLibOntologyResolver:
    graph = Graph()
    for class_name in (
        "RiskMetric",
        "CreditExposure",
        "Customer",
        "EnterpriseCustomer",
        "Subscription",
    ):
        graph.add((NS[class_name], RDF.type, OWL.Class))
    graph.add((NS.CreditExposure, RDFS.subClassOf, NS.RiskMetric))
    graph.add((NS.EnterpriseCustomer, RDFS.subClassOf, NS.Customer))
    graph.add((NS.RiskMetric, RDFS.subClassOf, OWL.Thing))
    graph.add((NS.hasSubscription, RDF.type, OWL.ObjectProperty))
    graph.add((NS.Customer, NS.hasSubscription, NS.Subscription))
    graph.add((NS.Acme, RDF.type, NS.EnterpriseCustomer))

    resolver = RDFLibOntologyResolver(ontology_file=None)
    resolver.graph = graph
    resolver.build_lookup()
    return resolver


def test_extract_candidate_terms_prefers_long_ngrams_and_drops_stopwords():
    terms = extract_candidate_terms("What is our credit exposure to the customer?")

    assert terms[0] == "credit_exposure"  # longest n-gram with content-word boundaries first
    assert "the_customer" not in terms  # stopword at an n-gram boundary
    assert "our" not in terms and "is" not in terms  # bare stopwords / short tokens
    assert "customer" in terms and "exposure" in terms


def test_ground_query_maps_terms_to_graph_node_ids_used_by_cognify():
    grounding = ground_query(
        "What's our credit exposure to the enterprise customer Acme?", _finance_resolver()
    )

    by_name = {concept.canonical_name: concept for concept in grounding.concepts}
    assert set(by_name) == {"CreditExposure", "EnterpriseCustomer", "Acme"}

    # Same deterministic ids construct_data_points_and_edges_with_ontology writes.
    assert by_name["CreditExposure"].node_id == str(EntityType.id_for("CreditExposure"))
    assert by_name["Acme"].node_id == str(Entity.id_for("Acme"))

    seeds = grounding.seed_node_ids_by_collection()
    assert set(seeds["EntityType_name"]) == {
        by_name["CreditExposure"].node_id,
        by_name["EnterpriseCustomer"].node_id,
    }
    assert seeds["Entity_name"] == [by_name["Acme"].node_id]


def test_ground_query_reports_is_a_chain_and_relations_with_canonical_casing():
    grounding = ground_query("credit exposure of Acme", _finance_resolver())
    by_name = {concept.canonical_name: concept for concept in grounding.concepts}

    assert by_name["CreditExposure"].parents == ("RiskMetric",)  # owl:Thing dropped
    assert by_name["Acme"].parents == ("EnterpriseCustomer", "Customer")  # owl:Class dropped

    block = grounding.to_context_block()
    assert block.startswith("## Ontology grounding")
    assert '"credit exposure" refers to CreditExposure (class); is a RiskMetric' in block
    assert "Acme (individual); is a EnterpriseCustomer > Customer" in block


def test_multiword_fuzzy_match_cannot_swallow_an_unrelated_token():
    # "enterprise customer acme" fuzzy-matches EnterpriseCustomer at >80%; accepting
    # it would consume "acme" and hide the individual. The token check rejects it,
    # so the class comes from the bigram and Acme still resolves on its own.
    grounding = ground_query("enterprise customer acme", _finance_resolver())

    terms = {concept.term: concept.canonical_name for concept in grounding.concepts}
    assert terms == {"enterprise customer": "EnterpriseCustomer", "acme": "Acme"}


def test_ground_query_is_empty_without_matches_or_resolver():
    resolver = _finance_resolver()

    assert not ground_query("hello there general", resolver)
    assert not ground_query("credit exposure", None)
    assert ground_query("", resolver) == QueryGrounding(query="")
    assert ground_query("hello", resolver).to_context_block() == ""
    assert ground_query("hello", resolver).seed_node_ids_by_collection() == {}


def test_ground_query_fails_open_on_resolver_errors():
    class BrokenResolver:
        def get_subgraph(self, node_name, node_type="individuals", directed=True):
            raise RuntimeError("boom")

    assert not ground_query("credit exposure", BrokenResolver())


def test_ground_query_respects_max_concepts():
    grounding = ground_query(
        "credit exposure, enterprise customer, subscription, Acme", _finance_resolver(), 2
    )
    assert len(grounding.concepts) == 2


def test_configured_entry_point_honours_explicit_flag_and_resolver(monkeypatch):
    resolver = _finance_resolver()

    assert ground_query_with_configured_ontology(
        "credit exposure", resolver=resolver, enabled=False
    ) == QueryGrounding(query="credit exposure")

    grounded = ground_query_with_configured_ontology("credit exposure", resolver=resolver)
    assert [concept.canonical_name for concept in grounded.concepts] == ["CreditExposure"]

    # No ontology configured -> nothing to ground against, even when asked to.
    from cognee.modules.ontology import query_grounding

    monkeypatch.setattr(query_grounding, "get_query_grounding_resolver", lambda: None)
    assert not ground_query_with_configured_ontology("credit exposure")
