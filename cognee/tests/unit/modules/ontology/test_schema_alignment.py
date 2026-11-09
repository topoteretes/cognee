"""Schema tables -> ontology classes: the first business-to-technical mapping."""

from uuid import uuid4

from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.modules.engine.models import EntityType
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
        authoritative_sources={"crm.customers": {"owner": "sales-ops"}},
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


def test_authority_lookup_accepts_every_value_shape():
    assert authority_for("crm.customers", {"customers": True}).authoritative is True
    assert authority_for("customers", {"customers": "sales-ops"}).owner == "sales-ops"
    assert (
        authority_for("customers", {"customers": {"authoritative": False}}).authoritative is False
    )
    assert authority_for("orders", {"customers": True}).authoritative is False
    assert authority_for("orders", None).authoritative is False


def test_env_authoritative_sources_parse_json_and_csv():
    assert parse_authoritative_sources('{"crm.customers": {"owner": "sales-ops"}}') == {
        "crm.customers": {"owner": "sales-ops"}
    }
    assert parse_authoritative_sources("crm.customers=sales-ops, orders") == {
        "crm.customers": {"authoritative": True, "owner": "sales-ops"},
        "orders": {"authoritative": True, "owner": None},
    }
    assert parse_authoritative_sources("{not json") == {}
    assert parse_authoritative_sources("") == {}
