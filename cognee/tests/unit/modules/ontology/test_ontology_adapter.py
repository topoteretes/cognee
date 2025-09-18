import pytest
from rdflib import Graph, Namespace, RDF, OWL, RDFS
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.ontology.get_default_ontology_resolver import get_default_ontology_resolver


def test_ontology_adapter_initialization_success():
    """Test successful initialization of OntologyAdapter."""

    adapter = get_default_ontology_resolver()
    adapter.build_lookup()

    assert isinstance(adapter.lookup, dict)


def test_ontology_adapter_initialization_file_not_found():
    """Test OntologyAdapter initialization with nonexistent file."""
    adapter = RDFLibOntologyResolver(ontology_file="nonexistent.owl")
    assert adapter.graph is None


def test_build_lookup():
    """Test the lookup dictionary is correctly built."""
    ns = Namespace("http://example.org/test#")
    g = Graph()

    g.add((ns.Car, RDF.type, OWL.Class))

    g.add((ns.Audi, RDF.type, ns.Car))

    resolver = RDFLibOntologyResolver()
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

    resolver = RDFLibOntologyResolver()
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

    resolver = RDFLibOntologyResolver()
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

    resolver = RDFLibOntologyResolver()
    resolver.graph = g
    resolver.build_lookup()

    result = resolver.find_closest_match("Nonexistent", "individuals")

    assert result is None


def test_get_subgraph_no_match_rdflib():
    """Test get_subgraph returns empty results for a non-existent node."""
    g = Graph()

    resolver = get_default_ontology_resolver()
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

    resolver = RDFLibOntologyResolver()
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

    config = get_default_ontology_resolver()
    resolver = config["resolver"]
    resolver.graph = g
    resolver.build_lookup()

    original_lookup = resolver.lookup

    resolver.refresh_lookup()

    assert resolver.lookup is not original_lookup


def test_fuzzy_matching_strategy_exact_match():
    """Test FuzzyMatchingStrategy finds exact matches."""
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    strategy = FuzzyMatchingStrategy()
    candidates = ["audi", "bmw", "mercedes"]

    result = strategy.find_match("audi", candidates)
    assert result == "audi"


def test_fuzzy_matching_strategy_fuzzy_match():
    """Test FuzzyMatchingStrategy finds fuzzy matches."""
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    strategy = FuzzyMatchingStrategy(cutoff=0.6)
    candidates = ["audi", "bmw", "mercedes"]

    result = strategy.find_match("audii", candidates)
    assert result == "audi"


def test_fuzzy_matching_strategy_no_match():
    """Test FuzzyMatchingStrategy returns None when no match meets cutoff."""
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    strategy = FuzzyMatchingStrategy(cutoff=0.9)
    candidates = ["audi", "bmw", "mercedes"]

    result = strategy.find_match("completely_different", candidates)
    assert result is None


def test_fuzzy_matching_strategy_empty_candidates():
    """Test FuzzyMatchingStrategy handles empty candidates list."""
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    strategy = FuzzyMatchingStrategy()

    result = strategy.find_match("audi", [])
    assert result is None


def test_base_ontology_resolver_initialization():
    """Test BaseOntologyResolver initialization with default matching strategy."""
    from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    class TestOntologyResolver(BaseOntologyResolver):
        def build_lookup(self):
            pass

        def refresh_lookup(self):
            pass

        def find_closest_match(self, name, category):
            return None

        def get_subgraph(self, node_name, node_type="individuals", directed=True):
            return [], [], None

    resolver = TestOntologyResolver()
    assert isinstance(resolver.matching_strategy, FuzzyMatchingStrategy)


def test_base_ontology_resolver_custom_matching_strategy():
    """Test BaseOntologyResolver initialization with custom matching strategy."""
    from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
    from cognee.modules.ontology.matching_strategies import MatchingStrategy

    class CustomMatchingStrategy(MatchingStrategy):
        def find_match(self, name, candidates):
            return "custom_match"

    class TestOntologyResolver(BaseOntologyResolver):
        def build_lookup(self):
            pass

        def refresh_lookup(self):
            pass

        def find_closest_match(self, name, category):
            return None

        def get_subgraph(self, node_name, node_type="individuals", directed=True):
            return [], [], None

    custom_strategy = CustomMatchingStrategy()
    resolver = TestOntologyResolver(matching_strategy=custom_strategy)
    assert resolver.matching_strategy == custom_strategy


def test_ontology_config_structure():
    """Test TypedDict structure for ontology configuration."""
    from cognee.modules.ontology.ontology_config import Config
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    matching_strategy = FuzzyMatchingStrategy()
    resolver = RDFLibOntologyResolver(matching_strategy=matching_strategy)

    config: Config = {"ontology_config": {"ontology_resolver": resolver}}

    assert config["ontology_config"]["ontology_resolver"] == resolver


def test_get_ontology_resolver_default():
    """Test get_ontology_resolver returns default configuration."""
    from cognee.modules.ontology.get_ontology_resolver import get_default_ontology_resolver
    from cognee.modules.ontology.ontology_config import Config
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    config: Config = get_default_ontology_resolver()

    assert isinstance(config["ontology_config"]["ontology_resolver"], RDFLibOntologyResolver)
    assert isinstance(
        config["ontology_config"]["ontology_resolver"].matching_strategy, FuzzyMatchingStrategy
    )


def test_get_default_ontology_resolver():
    """Test get_default_ontology_resolver returns default configuration."""
    from cognee.modules.ontology.get_ontology_resolver import get_default_ontology_resolver
    from cognee.modules.ontology.ontology_config import Config
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    config: Config = get_default_ontology_resolver()

    assert isinstance(config["ontology_config"]["ontology_resolver"], RDFLibOntologyResolver)
    assert isinstance(
        config["ontology_config"]["ontology_resolver"].matching_strategy, FuzzyMatchingStrategy
    )


def test_rdflib_ontology_resolver_uses_matching_strategy():
    """Test that RDFLibOntologyResolver uses the provided matching strategy."""
    from cognee.modules.ontology.matching_strategies import MatchingStrategy

    class TestMatchingStrategy(MatchingStrategy):
        def find_match(self, name, candidates):
            return "test_match" if candidates else None

    ns = Namespace("http://example.org/test#")
    g = Graph()
    g.add((ns.Car, RDF.type, OWL.Class))
    g.add((ns.Audi, RDF.type, ns.Car))

    resolver = RDFLibOntologyResolver(matching_strategy=TestMatchingStrategy())
    resolver.graph = g
    resolver.build_lookup()

    result = resolver.find_closest_match("Audi", "individuals")
    assert result == "test_match"


def test_rdflib_ontology_resolver_default_matching_strategy():
    """Test that RDFLibOntologyResolver uses FuzzyMatchingStrategy by default."""
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    resolver = RDFLibOntologyResolver()
    assert isinstance(resolver.matching_strategy, FuzzyMatchingStrategy)
