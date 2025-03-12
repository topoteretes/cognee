import pytest
from owlready2 import get_ontology, Thing
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver


def test_ontology_adapter_initialization_success():
    """Test successful initialization of OntologyAdapter."""
    ontology = get_ontology("http://example.org/test_ontology")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    assert adapter.ontology is not None
    assert isinstance(adapter.lookup, dict)


def test_ontology_adapter_initialization_file_not_found():
    """Test OntologyAdapter initialization with nonexistent file."""
    adapter = OntologyResolver(ontology_file="nonexistent.owl")
    assert adapter.ontology.base_iri == "http://example.org/empty_ontology#"


def test_build_lookup():
    """Test the lookup dictionary is correctly built."""
    ontology = get_ontology("http://example.org/test_ontology")

    with ontology:

        class Car(Thing):
            pass

        Car("Audi")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    assert isinstance(adapter.lookup, dict)
    assert "car" in adapter.lookup["classes"]
    assert "audi" in adapter.lookup["individuals"]


def test_find_closest_match_exact():
    """Test finding exact match in lookup."""
    ontology = get_ontology("http://example.org/test_ontology")

    with ontology:

        class Car(Thing):
            pass

        Car("Audi")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    result = adapter.find_closest_match("Audi", "individuals")

    assert result is not None
    assert result == "audi"


def test_find_closest_match_fuzzy():
    """Test fuzzy matching for lookup."""
    ontology = get_ontology("http://example.org/test_ontology")

    with ontology:

        class Car(Thing):
            pass

        Car("Audi")
        Car("BMW")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    result = adapter.find_closest_match("Audii", "individuals")

    assert result == "audi"


def test_find_closest_match_no_match():
    """Test no match found in lookup."""
    ontology = get_ontology("http://example.org/test_ontology")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    result = adapter.find_closest_match("Nonexistent", "individuals")

    assert result is None


def test_get_subgraph_no_match():
    """Test get_subgraph with no matching node."""
    ontology = get_ontology("http://example.org/test_ontology")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    nodes, relationships, start_node = adapter.get_subgraph("Nonexistent", "individuals")

    assert nodes == []
    assert relationships == []
    assert start_node is None


def test_get_subgraph_success():
    """Test successful retrieval of subgraph."""
    ontology = get_ontology("http://example.org/test_ontology")

    with ontology:

        class Company(Thing):
            pass

        class Vehicle(Thing):
            pass

        class Car(Vehicle):
            pass

        audi = Car("Audi")
        porsche = Car("Porsche")
        vw = Company("VW")

        vw.owns = [audi, porsche]

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    nodes, relationships, start_node = adapter.get_subgraph("Audi", "individuals")

    assert audi in nodes
    assert Car in nodes
    assert Vehicle in nodes
    assert Thing in nodes
    assert ("Audi", "is_a", "Car") in relationships
    assert ("Car", "is_a", "Vehicle") in relationships
    assert ("Vehicle", "is_a", "Thing") in relationships


def test_refresh_lookup():
    """Test refreshing lookup rebuilds the dictionary."""
    ontology = get_ontology("http://example.org/test_ontology")

    adapter = OntologyResolver()
    adapter.ontology = ontology
    adapter.build_lookup()

    original_lookup = adapter.lookup.copy()
    adapter.refresh_lookup()

    assert adapter.lookup is not original_lookup
