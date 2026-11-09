"""Schema tables -> ontology classes: the first business-to-technical mapping."""

import importlib
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.modules.engine.models import EntityType
from cognee.modules.ontology.exceptions import GetSubgraphError, OntologyInitializationError
from cognee.modules.ontology.ontology_env_config import OntologyEnvConfig
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.schema_alignment import (
    REALIZES_RELATIONSHIP,
    align_tables_with_ontology,
)

NS = Namespace("http://example.org/crm#")


def _crm_resolver() -> RDFLibOntologyResolver:
    graph = Graph()
    for class_name in ("Party", "Customer", "Company", "Subscription"):
        graph.add((NS[class_name], RDF.type, OWL.Class))
    graph.add((NS.Customer, RDFS.subClassOf, NS.Party))
    graph.add((NS.hasSubscription, RDF.type, OWL.ObjectProperty))
    graph.add((NS.Customer, NS.hasSubscription, NS.Subscription))

    resolver = RDFLibOntologyResolver(ontology_file=None)
    resolver.graph = graph
    resolver.build_lookup()
    return resolver


def _realizes(edges) -> dict:
    """Table node id -> properties of its ``realizes`` edge."""
    return {source: props for source, _, name, props in edges if name == REALIZES_RELATIONSHIP}


def test_a_table_realizes_the_class_its_name_is_the_exact_key_of():
    resolver = _crm_resolver()
    for class_name in ("Product", "Order", "Address", "CustomerAccount"):
        resolver.graph.add((NS[class_name], RDF.type, OWL.Class))
    resolver.build_lookup()
    names = ["public.companies", "addresses", "customer_accounts", "production", "borders"]
    table_names = {uuid4(): name for name in names}

    entity_types, edges = align_tables_with_ontology(table_names, resolver)

    type_names = {entity_type.id: entity_type.name for entity_type in entity_types}
    realized = {
        table_names[source]: type_names[target]
        for source, target, name, _ in edges
        if name == REALIZES_RELATIONSHIP
    }
    # A near miss is no match: "production" is not a Product, "borders" not an Order.
    assert realized == {
        "public.companies": "company",
        "addresses": "address",
        "customer_accounts": "customeraccount",
    }


def test_align_tables_emits_realizes_edges_and_the_ontology_subgraph():
    customers_id, companies_id, unknown_id = uuid4(), uuid4(), uuid4()

    entity_types, edges = align_tables_with_ontology(
        {customers_id: "public.customers", companies_id: "companies", unknown_id: "zz_audit"},
        _crm_resolver(),
    )

    triples = {(source, target, name) for source, target, name, _ in edges}
    customer_type_id = EntityType.id_for("Customer")
    assert (customers_id, customer_type_id, REALIZES_RELATIONSHIP) in triples
    assert (companies_id, EntityType.id_for("Company"), REALIZES_RELATIONSHIP) in triples
    # The class's own ontology edges ride along, exactly as cognify writes them.
    assert (customer_type_id, EntityType.id_for("Party"), "is_a") in triples
    assert (customer_type_id, EntityType.id_for("Subscription"), "hassubscription") in triples
    assert not any(source == unknown_id for source, _, _ in triples)

    # EntityType nodes carry ontology provenance under cognify's deterministic ids.
    types_by_id = {entity_type.id: entity_type for entity_type in entity_types}
    assert {entity_type.name for entity_type in entity_types} == {
        "customer",
        "party",
        "subscription",
        "company",
    }
    assert types_by_id[customer_type_id].ontology_valid is True
    assert types_by_id[customer_type_id].ontology_uri == str(NS.Customer)

    # The attribute dict matches the migration edge-tuple contract.
    for source, target, name, attributes in edges:
        assert attributes["source_node_id"] == str(source)
        assert attributes["target_node_id"] == str(target)
        assert attributes["relationship_name"] == name


def test_align_tables_dedupes_shared_ontology_edges_across_tables():
    _, edges = align_tables_with_ontology(
        {uuid4(): "crm.customers", uuid4(): "legacy.customer"}, _crm_resolver()
    )
    assert len([edge for edge in edges if edge[2] == "is_a"]) == 1  # once, not once per table
    assert len(_realizes(edges)) == 2


def test_align_tables_is_empty_without_resolver_or_tables():
    assert align_tables_with_ontology({uuid4(): "customers"}, None) == ([], [])
    assert align_tables_with_ontology({}, _crm_resolver()) == ([], [])
    assert align_tables_with_ontology({uuid4(): "zz_audit"}, _crm_resolver()) == ([], [])


# --- authority ----------------------------------------------------------------------

from cognee.modules.ontology.ontology_env_config import parse_authoritative_sources  # noqa: E402
from cognee.modules.ontology.schema_alignment import authority_for  # noqa: E402


def test_authoritative_sources_are_stamped_on_realizes_edges():
    crm_id, legacy_id = uuid4(), uuid4()
    _, edges = align_tables_with_ontology(
        {crm_id: "crm.customers", legacy_id: "legacy.customer"},
        _crm_resolver(),
        authoritative_sources={"crm.customers": "sales-ops"},
    )
    realizes = _realizes(edges)
    assert realizes[crm_id]["authoritative"] is True
    assert realizes[crm_id]["owner"] == "sales-ops"
    assert realizes[crm_id]["ontology_valid"] is True
    assert realizes[legacy_id]["authoritative"] is False
    assert realizes[legacy_id]["owner"] is None


def test_a_listed_table_is_authoritative_by_qualified_or_bare_name():
    assert authority_for("crm.customers", {"customers": None}) == (True, None)
    assert authority_for("customers", {"customers": "sales-ops"}) == (True, "sales-ops")
    assert authority_for("orders", {"customers": None}) == (False, None)
    assert authority_for("orders", None) == (False, None)


def test_env_authoritative_sources_parse_the_csv_and_reject_a_missing_table():
    assert parse_authoritative_sources("crm.customers=sales-ops, orders") == {
        "crm.customers": "sales-ops",
        "orders": None,
    }
    assert parse_authoritative_sources("") == {}
    with pytest.raises(ValueError):
        parse_authoritative_sources("=x")


def test_a_per_call_empty_map_overrides_the_environment(monkeypatch):
    from cognee.modules.ontology import get_default_ontology_resolver as resolver_module

    config = OntologyEnvConfig(ontology_authoritative_sources="crm.customers=sales-ops")
    monkeypatch.setattr(resolver_module, "get_ontology_env_config", lambda: config)
    per_call = resolver_module.get_configured_authoritative_sources(
        {"ontology_config": {"authoritative_sources": {}}}
    )
    _, edges = align_tables_with_ontology(
        {uuid4(): "crm.customers"}, _crm_resolver(), authoritative_sources=per_call
    )

    assert [props["authoritative"] for props in _realizes(edges).values()] == [False]
    assert resolver_module.get_configured_authoritative_sources() == {"crm.customers": "sales-ops"}


# --- errors ---------------------------------------------------------------------------


def test_a_failing_ontology_lookup_fails_the_alignment():
    resolver = _crm_resolver()

    def broken_get_subgraph(*_args, **_kwargs):
        raise GetSubgraphError()

    resolver.get_subgraph = broken_get_subgraph

    with pytest.raises(GetSubgraphError):
        align_tables_with_ontology({uuid4(): "customers"}, resolver)


@pytest.mark.asyncio
async def test_a_malformed_ontology_file_fails_the_migration(monkeypatch, tmp_path):
    from cognee.modules.ontology import get_default_ontology_resolver as resolver_module

    migration = importlib.import_module("cognee.tasks.ingestion.migrate_relational_database")
    malformed = tmp_path / "ontology.owl"
    malformed.write_text("this is <not rdf")
    config = OntologyEnvConfig(ontology_file_path=str(malformed))
    monkeypatch.setattr(resolver_module, "get_ontology_env_config", lambda: config)
    monkeypatch.setattr(migration, "schema_only_ingestion", AsyncMock(return_value=({}, [])))

    with pytest.raises(OntologyInitializationError):
        await migration.migrate_relational_database(MagicMock(), {}, schema_only=True)


@pytest.mark.asyncio
async def test_migration_uses_the_given_resolver_and_sources(monkeypatch):
    from cognee.modules.engine.models import TableType

    migration = importlib.import_module("cognee.tasks.ingestion.migrate_relational_database")
    table = TableType(id=uuid4(), name="customers", description="customers table")
    monkeypatch.setattr(
        migration, "schema_only_ingestion", AsyncMock(return_value=({"customers": table}, []))
    )
    monkeypatch.setattr(migration, "index_data_points", AsyncMock())
    monkeypatch.setattr(migration, "index_graph_edges", AsyncMock())
    graph_db = MagicMock(add_nodes=AsyncMock(), add_edges=AsyncMock(), get_graph_data=AsyncMock())

    await migration.migrate_relational_database(
        graph_db,
        {},
        schema_only=True,
        ontology_resolver=_crm_resolver(),
        authoritative_sources={"customers": "sales-ops"},
    )

    edges = graph_db.add_edges.await_args.args[0]
    realizes = [props for source, _, name, props in edges if name == REALIZES_RELATIONSHIP]
    assert [(props["authoritative"], props["owner"]) for props in realizes] == [(True, "sales-ops")]
