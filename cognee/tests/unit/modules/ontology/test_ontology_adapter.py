import pytest
from rdflib import Graph, Namespace, RDF, OWL, RDFS
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver, AttachedOntologyNode


def test_ontology_adapter_initialization_success():
    """Test successful initialization of OntologyAdapter."""

    adapter = OntologyResolver()
    adapter.build_lookup()

    assert isinstance(adapter.lookup, dict)


def test_ontology_adapter_initialization_file_not_found():
    """Test OntologyAdapter initialization with nonexistent file."""
    adapter = OntologyResolver(ontology_file="nonexistent.owl")
    assert adapter.graph is None


def test_build_lookup():
    """Test the lookup dictionary is correctly built."""
    ns = Namespace("http://example.org/test#")
    g = Graph()

    g.add((ns.Car, RDF.type, OWL.Class))

    g.add((ns.Audi, RDF.type, ns.Car))

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    lookup = resolver.lookup
    assert isinstance(lookup, dict)

    assert "car" in lookup["classes"]
    assert lookup["classes"]["car"] == ns.Car

    assert "audi" in lookup["individuals"]
    assert lookup["individuals"]["audi"] == ns.Audi


def test_find_closest_match_exact():
    """Test finding exact match in lookup."""

    ns = Namespace("http://example.org/test#")
    g = Graph()

    g.add((ns.Car, RDF.type, OWL.Class))
    g.add((ns.Audi, RDF.type, ns.Car))

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    result = resolver.find_closest_match("Audi", "individuals")
    assert result is not None
    assert result == "audi"


def test_find_closest_match_fuzzy():
    """Test fuzzy matching for lookup using the RDFlib adapter."""

    ns = Namespace("http://example.org/test#")

    g = Graph()

    g.add((ns.Car, RDF.type, OWL.Class))

    g.add((ns.Audi, RDF.type, ns.Car))
    g.add((ns.BMW, RDF.type, ns.Car))

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    result = resolver.find_closest_match("Audii", "individuals")

    assert result == "audi"


def test_find_closest_match_no_match():
    """Test no match found in lookup."""
    """Test that find_closest_match returns None when there is no match."""
    ns = Namespace("http://example.org/test#")

    g = Graph()

    g.add((ns.Car, RDF.type, OWL.Class))

    g.add((ns.Audi, RDF.type, ns.Car))
    g.add((ns.BMW, RDF.type, ns.Car))

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    result = resolver.find_closest_match("Nonexistent", "individuals")

    assert result is None


def test_get_subgraph_no_match_rdflib():
    """Test get_subgraph returns empty results for a non-existent node."""
    g = Graph()

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    nodes, relationships, start_node = resolver.get_subgraph("Nonexistent", "individuals")

    assert nodes == []
    assert relationships == []
    assert start_node is None


def test_get_subgraph_success_rdflib():
    """Test successful retrieval of subgraph using the RDFlib adapter."""

    ns = Namespace("http://example.org/test#")
    g = Graph()

    g.add((ns.Company, RDF.type, OWL.Class))
    g.add((ns.Vehicle, RDF.type, OWL.Class))
    g.add((ns.Car, RDF.type, OWL.Class))

    g.add((ns.Vehicle, RDFS.subClassOf, OWL.Thing))
    g.add((ns.Car, RDFS.subClassOf, ns.Vehicle))

    g.add((ns.Audi, RDF.type, ns.Car))
    g.add((ns.Porsche, RDF.type, ns.Car))
    g.add((ns.VW, RDF.type, ns.Company))

    owns = ns.owns
    g.add((owns, RDF.type, OWL.ObjectProperty))
    g.add((ns.VW, owns, ns.Audi))
    g.add((ns.VW, owns, ns.Porsche))

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    nodes, relationships, start_node = resolver.get_subgraph("Audi", "individuals")

    uris = {n.uri for n in nodes}
    assert ns.Audi in uris
    assert ns.Car in uris
    assert ns.Vehicle in uris
    assert OWL.Thing in uris

    rels = set(relationships)
    assert ("audi", "is_a", "car") in rels
    assert ("car", "is_a", "vehicle") in rels
    assert ("vehicle", "is_a", "thing") in rels

    assert isinstance(start_node, AttachedOntologyNode)
    assert start_node.uri == ns.Audi


def test_refresh_lookup_rdflib():
    """Test that refresh_lookup rebuilds the lookup dict into a new object."""
    g = Graph()

    resolver = OntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    original_lookup = resolver.lookup

    resolver.refresh_lookup()

    assert resolver.lookup is not original_lookup
