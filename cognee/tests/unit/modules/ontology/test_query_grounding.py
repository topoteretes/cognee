"""Read-time ontology grounding: query terms -> ontology nodes -> seeds + context."""

import os
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology import matching_strategies, query_grounding
from cognee.modules.ontology.exceptions import GetSubgraphError, OntologyInitializationError
from cognee.modules.ontology.ontology_env_config import OntologyEnvConfig
from cognee.modules.ontology.query_grounding import (
    ground_query,
    grounding_block,
    seed_node_ids_by_collection,
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
        "LineOfCredit",
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


def test_ground_query_maps_terms_to_graph_node_ids_used_by_cognify():
    grounding = ground_query(
        "What's our credit exposure to the enterprise customer Acme?", _finance_resolver()
    )

    by_name = {concept.canonical_name: concept for concept in grounding}
    assert set(by_name) == {"CreditExposure", "EnterpriseCustomer", "Acme"}

    # Same deterministic ids construct_data_points_and_edges_with_ontology writes.
    assert by_name["CreditExposure"].node_id == str(EntityType.id_for("CreditExposure"))
    assert by_name["Acme"].node_id == str(Entity.id_for("Acme"))

    seeds = seed_node_ids_by_collection(grounding)
    assert set(seeds["EntityType_name"]) == {
        by_name["CreditExposure"].node_id,
        by_name["EnterpriseCustomer"].node_id,
    }
    assert seeds["Entity_name"] == [by_name["Acme"].node_id]


def test_ground_query_reports_is_a_chain_and_relations_with_canonical_casing():
    grounding = ground_query("credit exposure of Acme", _finance_resolver())
    by_name = {concept.canonical_name: concept for concept in grounding}

    assert by_name["CreditExposure"].parents == ("RiskMetric",)  # owl:Thing dropped
    assert by_name["Acme"].parents == ("EnterpriseCustomer", "Customer")  # owl:Class dropped

    block = grounding_block(grounding)
    assert block.startswith("## Ontology grounding")
    assert '"credit exposure" refers to CreditExposure (class); is a RiskMetric' in block
    assert "Acme (individual); is a EnterpriseCustomer > Customer" in block


def test_a_longer_run_cannot_swallow_an_unrelated_word():
    # "enterprise customer acme" is no key, so it cannot consume "acme": the class
    # comes from the bigram and Acme still resolves on its own.
    grounding = ground_query("enterprise customer acme", _finance_resolver())

    terms = {concept.term: concept.canonical_name for concept in grounding}
    assert terms == {"enterprise customer": "EnterpriseCustomer", "acme": "Acme"}


def test_ground_query_is_empty_without_matches_or_resolver():
    resolver = _finance_resolver()

    assert not ground_query("hello there general", resolver)
    assert not ground_query("credit exposure", None)
    assert ground_query("", resolver) == []
    assert grounding_block(ground_query("hello", resolver)) == ""
    assert seed_node_ids_by_collection(ground_query("hello", resolver)) == {}


def test_ground_query_raises_resolver_errors():
    class BrokenResolver:
        lookup = {"classes": {"creditexposure": NS.CreditExposure}, "individuals": {}}

        def get_subgraph(self, node_name, node_type="individuals", directed=True):
            raise GetSubgraphError()

    with pytest.raises(GetSubgraphError):
        ground_query("credit exposure", BrokenResolver())


def test_grounding_matches_exact_keys_without_a_fuzzy_scan(monkeypatch):
    resolver = _finance_resolver()
    for index in range(5000):
        resolver.graph.add((NS[f"Generated{index}"], RDF.type, OWL.Class))
    resolver.build_lookup()

    def no_fuzzy_scan(*_args, **_kwargs):
        raise AssertionError("difflib.get_close_matches was called")

    monkeypatch.setattr(matching_strategies.difflib, "get_close_matches", no_fuzzy_scan)
    grounding = ground_query(
        "Which table stores the credit exposures of our enterprise customers, and who owns "
        "the customer accounts data in the sales operations team?",
        resolver,
    )

    assert [concept.canonical_name for concept in grounding] == [
        "CreditExposure",
        "EnterpriseCustomer",
        "Customer",
    ]


def test_grounding_keeps_inner_stopwords_and_matches_each_occurrence():
    resolver = _finance_resolver()

    line_of_credit = ground_query("what is the line of credit for Acme?", resolver)
    assert [c.canonical_name for c in line_of_credit] == ["LineOfCredit", "Acme"]

    # The first "customer" is used up by "enterprise customer"; the second still grounds.
    twice = ground_query("which enterprise customer is our largest customer?", resolver)
    assert [c.canonical_name for c in twice] == ["EnterpriseCustomer", "Customer"]


def test_a_name_may_start_or_end_with_a_stopword_but_a_lone_stopword_is_no_name():
    resolver = _finance_resolver()
    for class_name in ("OffBalanceSheetExposure", "NewCustomer", "Will", "City"):
        resolver.graph.add((NS[class_name], RDF.type, OWL.Class))
    resolver.graph.add((NS.NewYork, RDF.type, NS.City))
    resolver.build_lookup()

    def grounded(query: str) -> list[str]:
        return [concept.canonical_name for concept in ground_query(query, resolver)]

    assert grounded("what is our off balance sheet exposure?") == ["OffBalanceSheetExposure"]
    assert grounded("how many new customers joined?") == ["NewCustomer"]
    assert grounded("customers in New York") == ["NewYork", "Customer"]
    # "will" alone is a verb here, not the class Will.
    assert grounded("who will sign it?") == []


# --- loading the configured resolver ------------------------------------------------


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """Point grounding at an ontology file in ``tmp_path``; returns a config setter."""
    query_grounding._load_resolver.cache_clear()
    path = tmp_path / "ontology.owl"

    def configure(**settings) -> Path:
        config = OntologyEnvConfig(ontology_file_path=str(path), **settings)
        monkeypatch.setattr(query_grounding, "get_ontology_env_config", lambda: config)
        return path

    yield configure
    query_grounding._load_resolver.cache_clear()


def _write_classes(path: Path, *class_names: str, mtime: int) -> None:
    graph = Graph()
    for class_name in class_names:
        graph.add((NS[class_name], RDF.type, OWL.Class))
    graph.serialize(str(path), format="xml")
    os.utime(path, (mtime, mtime))


def test_configured_resolver_is_built_with_every_setting(configured, monkeypatch):
    configured(ontology_reasoning="owl_rl")
    built = []
    monkeypatch.setattr(
        query_grounding, "get_ontology_resolver_from_env", lambda **kwargs: built.append(kwargs)
    )

    query_grounding.get_query_grounding_resolver()

    assert built[0]["ontology_reasoning"] == "owl_rl"


def test_a_load_error_is_raised_and_not_cached(configured, monkeypatch):
    configured()
    resolver = object()
    outcomes = [OntologyInitializationError("half-written file"), resolver]

    def factory(**_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(query_grounding, "get_ontology_resolver_from_env", factory)

    with pytest.raises(OntologyInitializationError):
        query_grounding.get_query_grounding_resolver()
    assert query_grounding.get_query_grounding_resolver() is resolver


def test_an_edited_ontology_file_is_loaded_again(configured):
    path = configured()
    _write_classes(path, "CreditExposure", mtime=1_000_000)
    first = query_grounding.get_query_grounding_resolver()
    assert not ground_query("loan facility", first)
    assert query_grounding.get_query_grounding_resolver() is first

    _write_classes(path, "CreditExposure", "LoanFacility", mtime=2_000_000)
    grounding = ground_query("loan facility", query_grounding.get_query_grounding_resolver())

    assert [concept.canonical_name for concept in grounding] == ["LoanFacility"]
