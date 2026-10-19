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


# --- columns, properties and authority --------------------------------------------

from cognee.modules.engine.models import OntologyProperty  # noqa: E402
from cognee.modules.ontology.ontology_env_config import parse_authoritative_sources  # noqa: E402
from cognee.modules.ontology.schema_alignment import (  # noqa: E402
    HAS_COLUMN_RELATIONSHIP,
    SchemaTableSpec,
    authority_for,
    column_lookup_names,
    column_node_id,
    column_specs_from,
)


def _crm_resolver_with_properties() -> RDFLibOntologyResolver:
    resolver = _crm_resolver()
    graph = resolver.graph
    graph.add((NS.hasCustomerId, RDF.type, OWL.DatatypeProperty))
    graph.add((NS.hasCustomerId, RDFS.domain, NS.Customer))
    graph.add((NS.hasEmail, RDF.type, OWL.DatatypeProperty))
    graph.add((NS.hasEmail, RDFS.domain, NS.Party))
    graph.add((NS.hasSubscription, RDFS.domain, NS.Customer))
    graph.add((NS.hasSubscription, RDFS.range, NS.Subscription))
    resolver.build_lookup()
    return resolver


def test_build_lookup_indexes_object_and_datatype_properties():
    resolver = _crm_resolver_with_properties()
    assert set(resolver.lookup["properties"]) == {"hascustomerid", "hasemail", "hassubscription"}
    _, edges, root = resolver.get_subgraph("has_customer_id", node_type="properties")
    assert root.name == "hasCustomerId"
    assert ("hascustomerid", "domain", "customer") in edges
    assert ("customer", "is_a", "party") in edges  # the domain's own chain rides along


def test_column_lookup_names_expand_abbreviated_id_columns():
    assert column_lookup_names("email") == ["email", "has_email"]
    assert column_lookup_names("cust_id", "Customer") == [
        "cust_id",
        "has_cust_id",
        "has_customer_id",
        "customer_id",
    ]
    # A stem that is not a prefix of the class does not get expanded.
    assert column_lookup_names("order_id", "Customer") == ["order_id", "has_order_id"]


def test_column_specs_accept_every_shape_the_ingestion_paths_carry():
    assert [c.name for c in column_specs_from([{"name": "id", "type": "INT"}])] == ["id"]
    assert [c.name for c in column_specs_from({"id": {"type": "INT"}})] == ["id"]
    assert [c.name for c in column_specs_from('[{"name": "id"}]')] == ["id"]
    assert [c.name for c in column_specs_from(["id", "email"])] == ["id", "email"]
    assert column_specs_from("not json") == ()


def test_columns_realize_properties_with_domain_check():
    customers_id, orders_id = uuid4(), uuid4()
    alignment = align_tables_with_ontology(
        [
            SchemaTableSpec(
                customers_id,
                "crm.customers",
                column_specs_from([{"name": "cust_id"}, {"name": "email"}, {"name": "zz"}]),
            ),
            # ``orders`` realizes no class; ``email`` (domain Party) is still accepted
            # because the table's class is unknown, ``cust_id`` cannot be expanded.
            SchemaTableSpec(orders_id, "orders", column_specs_from(["email", "cust_id"])),
        ],
        _crm_resolver_with_properties(),
    )

    assert alignment.realized_property_by_column == {
        "crm.customers.cust_id": "hasCustomerId",
        "crm.customers.email": "hasEmail",
        "orders.email": "hasEmail",
    }
    edges = {(str(s), str(t), n) for s, t, n, _ in alignment.edges}
    cust_col = str(column_node_id(customers_id, "cust_id"))
    prop_id = str(OntologyProperty.id_for("hasCustomerId"))
    assert (str(customers_id), cust_col, HAS_COLUMN_RELATIONSHIP) in edges
    assert (cust_col, prop_id, REALIZES_RELATIONSHIP) in edges
    assert (prop_id, str(EntityType.id_for("Customer")), "domain") in edges
    column = alignment.columns[column_node_id(customers_id, "cust_id")]
    assert column.table_name == "crm.customers"
    assert alignment.properties[OntologyProperty.id_for("hasCustomerId")].ontology_valid is True


def test_property_whose_domain_misses_the_table_class_is_skipped():
    subscriptions_id = uuid4()
    alignment = align_tables_with_ontology(
        [SchemaTableSpec(subscriptions_id, "subscriptions", column_specs_from(["email"]))],
        _crm_resolver_with_properties(),
    )
    # hasEmail's domain is Party; Subscription is not a Party.
    assert alignment.realized_property_by_column == {}
    assert alignment.realized_concept_by_table == {"subscriptions": "Subscription"}


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
