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
    table_lookup_names,
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


def test_table_lookup_names_strips_schema_and_singularizes():
    assert table_lookup_names("public.companies") == ["companies", "company"]
    assert table_lookup_names("customers") == ["customers", "customer"]
    assert table_lookup_names("addresses") == ["addresses", "address"]
    assert table_lookup_names("status") == ["status"]  # -ss/-us words are left alone


def test_align_tables_emits_realizes_edges_and_the_ontology_subgraph():
    customers_id, companies_id, unknown_id = uuid4(), uuid4(), uuid4()

    alignment = align_tables_with_ontology(
        [(customers_id, "public.customers"), (companies_id, "companies"), (unknown_id, "zz_audit")],
        _crm_resolver(),
    )

    assert alignment.realized_concept_by_table == {
        "public.customers": "Customer",
        "companies": "Company",
    }

    edges = {(str(source), str(target), name) for source, target, name, _ in alignment.edges}
    customer_type_id = str(EntityType.id_for("Customer"))
    assert (str(customers_id), customer_type_id, REALIZES_RELATIONSHIP) in edges
    assert (str(companies_id), str(EntityType.id_for("Company")), REALIZES_RELATIONSHIP) in edges
    # The class's own ontology edges ride along, exactly as cognify writes them.
    assert (customer_type_id, str(EntityType.id_for("Party")), "is_a") in edges
    assert (customer_type_id, str(EntityType.id_for("Subscription")), "hassubscription") in edges
    assert not any(str(unknown_id) == source for source, _, _ in edges)

    # EntityType nodes carry ontology provenance under cognify's deterministic ids.
    names = {entity_type.name for entity_type in alignment.entity_types.values()}
    assert names == {"customer", "party", "subscription", "company"}
    customer_type = alignment.entity_types[EntityType.id_for("Customer")]
    assert customer_type.ontology_valid is True
    assert customer_type.ontology_uri == str(NS.Customer)

    # Edge attribute dict matches the migration edge-tuple contract.
    _, _, _, attributes = alignment.edges[0]
    assert set(attributes) == {"source_node_id", "target_node_id", "relationship_name"}


def test_align_tables_dedupes_shared_ontology_edges_across_tables():
    alignment = align_tables_with_ontology(
        [(uuid4(), "crm.customers"), (uuid4(), "legacy.customer")], _crm_resolver()
    )
    is_a_edges = [edge for edge in alignment.edges if edge[2] == "is_a"]
    assert len(is_a_edges) == 1  # Customer is_a Party appears once, not once per table
    assert len([edge for edge in alignment.edges if edge[2] == REALIZES_RELATIONSHIP]) == 2


def test_align_tables_is_empty_without_resolver_or_tables():
    assert not align_tables_with_ontology([(uuid4(), "customers")], None)
    assert not align_tables_with_ontology([], _crm_resolver())
    assert not align_tables_with_ontology([(uuid4(), "zz_audit")], _crm_resolver())


# --- authority ----------------------------------------------------------------------

from cognee.modules.ontology.ontology_env_config import parse_authoritative_sources  # noqa: E402
from cognee.modules.ontology.schema_alignment import authority_for  # noqa: E402


def test_authoritative_sources_are_stamped_on_realizes_edges():
    crm_id, legacy_id = uuid4(), uuid4()
    alignment = align_tables_with_ontology(
        [(crm_id, "crm.customers"), (legacy_id, "legacy.customer")],
        _crm_resolver(),
        authoritative_sources={"crm.customers": "sales-ops"},
    )
    realizes = {
        str(source): props
        for source, _, name, props in alignment.edges
        if name == REALIZES_RELATIONSHIP
    }
    assert realizes[str(crm_id)]["authoritative"] is True
    assert realizes[str(crm_id)]["owner"] == "sales-ops"
    assert realizes[str(crm_id)]["ontology_valid"] is True
    assert realizes[str(legacy_id)]["authoritative"] is False
    assert realizes[str(legacy_id)]["owner"] is None


def test_a_listed_table_is_authoritative_by_qualified_or_bare_name():
    assert authority_for("crm.customers", {"customers": None}).authoritative is True
    assert authority_for("customers", {"customers": "sales-ops"}).owner == "sales-ops"
    assert authority_for("orders", {"customers": None}).authoritative is False
    assert authority_for("orders", None).authoritative is False


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
    alignment = align_tables_with_ontology(
        [(uuid4(), "crm.customers")], _crm_resolver(), authoritative_sources=per_call
    )

    assert [
        props["authoritative"]
        for _, _, name, props in alignment.edges
        if name == REALIZES_RELATIONSHIP
    ] == [False]
    assert resolver_module.get_configured_authoritative_sources() == {"crm.customers": "sales-ops"}


# --- errors ---------------------------------------------------------------------------


def test_a_failing_ontology_lookup_fails_the_alignment():
    class BrokenResolver(RDFLibOntologyResolver):
        def get_subgraph(self, node_name, node_type="individuals", directed=True):
            raise GetSubgraphError()

    with pytest.raises(GetSubgraphError):
        align_tables_with_ontology([(uuid4(), "customers")], BrokenResolver(ontology_file=None))


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
