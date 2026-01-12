import pytest
from rdflib import Graph, Namespace, RDF, OWL, RDFS
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.ontology.get_default_ontology_resolver import get_default_ontology_resolver


def test_ontology_adapter_initialization_success():
    """Test successful initialization of RDFLibOntologyResolver from get_default_ontology_resolver."""

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
    """Test get_subgraph returns empty results for a non-existent node using RDFLibOntologyResolver."""
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
    """Test that refresh_lookup rebuilds the lookup dict into a new object using RDFLibOntologyResolver."""
    g = Graph()

    resolver = get_default_ontology_resolver()
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
    """Test get_default_ontology_resolver returns a properly configured RDFLibOntologyResolver with FuzzyMatchingStrategy."""
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    resolver = get_default_ontology_resolver()

    assert isinstance(resolver, RDFLibOntologyResolver)
    assert isinstance(resolver.matching_strategy, FuzzyMatchingStrategy)


def test_get_default_ontology_resolver():
    """Test get_default_ontology_resolver returns a properly configured RDFLibOntologyResolver with FuzzyMatchingStrategy."""
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    resolver = get_default_ontology_resolver()

    assert isinstance(resolver, RDFLibOntologyResolver)
    assert isinstance(resolver.matching_strategy, FuzzyMatchingStrategy)


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


def test_get_ontology_resolver_from_env_success():
    """Test get_ontology_resolver_from_env returns correct resolver with valid parameters."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    resolver = get_ontology_resolver_from_env(
        ontology_resolver="rdflib", matching_strategy="fuzzy", ontology_file_path="/test/path.owl"
    )

    assert isinstance(resolver, RDFLibOntologyResolver)
    assert isinstance(resolver.matching_strategy, FuzzyMatchingStrategy)
    assert resolver.ontology_file == "/test/path.owl"


def test_get_ontology_resolver_from_env_unsupported_resolver():
    """Test get_ontology_resolver_from_env raises EnvironmentError for unsupported resolver."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env(
            ontology_resolver="unsupported",
            matching_strategy="fuzzy",
            ontology_file_path="/test/path.owl",
        )

    assert "Unsupported ontology resolver: unsupported" in str(exc_info.value)
    assert "Supported resolvers are: RdfLib with FuzzyMatchingStrategy" in str(exc_info.value)


def test_get_ontology_resolver_from_env_unsupported_strategy():
    """Test get_ontology_resolver_from_env raises EnvironmentError for unsupported strategy."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env(
            ontology_resolver="rdflib",
            matching_strategy="unsupported",
            ontology_file_path="/test/path.owl",
        )

    assert "Unsupported ontology resolver: rdflib" in str(exc_info.value)


def test_get_ontology_resolver_from_env_empty_file_path():
    """Test get_ontology_resolver_from_env raises EnvironmentError for empty file path."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env(
            ontology_resolver="rdflib", matching_strategy="fuzzy", ontology_file_path=""
        )

    assert "Unsupported ontology resolver: rdflib" in str(exc_info.value)


def test_get_ontology_resolver_from_env_none_file_path():
    """Test get_ontology_resolver_from_env raises EnvironmentError for None file path."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env(
            ontology_resolver="rdflib", matching_strategy="fuzzy", ontology_file_path=None
        )

    assert "Unsupported ontology resolver: rdflib" in str(exc_info.value)


def test_get_ontology_resolver_from_env_empty_resolver():
    """Test get_ontology_resolver_from_env raises EnvironmentError for empty resolver."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env(
            ontology_resolver="", matching_strategy="fuzzy", ontology_file_path="/test/path.owl"
        )

    assert "Unsupported ontology resolver:" in str(exc_info.value)


def test_get_ontology_resolver_from_env_empty_strategy():
    """Test get_ontology_resolver_from_env raises EnvironmentError for empty strategy."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env(
            ontology_resolver="rdflib", matching_strategy="", ontology_file_path="/test/path.owl"
        )

    assert "Unsupported ontology resolver: rdflib" in str(exc_info.value)


def test_get_ontology_resolver_from_env_default_parameters():
    """Test get_ontology_resolver_from_env with default empty parameters raises EnvironmentError."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError) as exc_info:
        get_ontology_resolver_from_env()

    assert "Unsupported ontology resolver:" in str(exc_info.value)


def test_get_ontology_resolver_from_env_case_sensitivity():
    """Test get_ontology_resolver_from_env is case sensitive."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    with pytest.raises(EnvironmentError):
        get_ontology_resolver_from_env(
            ontology_resolver="RDFLIB",
            matching_strategy="fuzzy",
            ontology_file_path="/test/path.owl",
        )

    with pytest.raises(EnvironmentError):
        get_ontology_resolver_from_env(
            ontology_resolver="RdfLib",
            matching_strategy="fuzzy",
            ontology_file_path="/test/path.owl",
        )


def test_get_ontology_resolver_from_env_with_actual_file():
    """Test get_ontology_resolver_from_env works with actual file path."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env
    from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
    from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy

    resolver = get_ontology_resolver_from_env(
        ontology_resolver="rdflib",
        matching_strategy="fuzzy",
        ontology_file_path="/path/to/ontology.owl",
    )

    assert isinstance(resolver, RDFLibOntologyResolver)
    assert isinstance(resolver.matching_strategy, FuzzyMatchingStrategy)
    assert resolver.ontology_file == "/path/to/ontology.owl"


def test_get_ontology_resolver_from_env_resolver_functionality():
    """Test that resolver created from env function works correctly."""
    from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env

    resolver = get_ontology_resolver_from_env(
        ontology_resolver="rdflib", matching_strategy="fuzzy", ontology_file_path="/test/path.owl"
    )

    resolver.build_lookup()
    assert isinstance(resolver.lookup, dict)

    result = resolver.find_closest_match("test", "individuals")
    assert result is None  # Should return None for non-existent entity

    nodes, relationships, start_node = resolver.get_subgraph("test", "individuals")
    assert nodes == []
    assert relationships == []
    assert start_node is None


def test_multifile_ontology_loading_success():
    """Test successful loading of multiple ontology files."""
    ns1 = Namespace("http://example.org/cars#")
    ns2 = Namespace("http://example.org/tech#")

    g1 = Graph()
    g1.add((ns1.Vehicle, RDF.type, OWL.Class))
    g1.add((ns1.Car, RDF.type, OWL.Class))
    g1.add((ns1.Car, RDFS.subClassOf, ns1.Vehicle))
    g1.add((ns1.Audi, RDF.type, ns1.Car))
    g1.add((ns1.BMW, RDF.type, ns1.Car))

    g2 = Graph()
    g2.add((ns2.Company, RDF.type, OWL.Class))
    g2.add((ns2.TechCompany, RDF.type, OWL.Class))
    g2.add((ns2.TechCompany, RDFS.subClassOf, ns2.Company))
    g2.add((ns2.Apple, RDF.type, ns2.TechCompany))
    g2.add((ns2.Google, RDF.type, ns2.TechCompany))

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False) as f1:
        g1.serialize(f1.name, format="xml")
        file1_path = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False) as f2:
        g2.serialize(f2.name, format="xml")
        file2_path = f2.name

    try:
        resolver = RDFLibOntologyResolver(ontology_file=[file1_path, file2_path])

        assert resolver.graph is not None

        assert "car" in resolver.lookup["classes"]
        assert "vehicle" in resolver.lookup["classes"]
        assert "company" in resolver.lookup["classes"]
        assert "techcompany" in resolver.lookup["classes"]

        assert "audi" in resolver.lookup["individuals"]
        assert "bmw" in resolver.lookup["individuals"]
        assert "apple" in resolver.lookup["individuals"]
        assert "google" in resolver.lookup["individuals"]

        car_match = resolver.find_closest_match("Audi", "individuals")
        assert car_match == "audi"

        tech_match = resolver.find_closest_match("Google", "individuals")
        assert tech_match == "google"

    finally:
        import os

        os.unlink(file1_path)
        os.unlink(file2_path)


def test_multifile_ontology_with_missing_files():
    """Test loading multiple ontology files where some don't exist."""
    ns = Namespace("http://example.org/test#")
    g = Graph()
    g.add((ns.Car, RDF.type, OWL.Class))
    g.add((ns.Audi, RDF.type, ns.Car))

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False) as f:
        g.serialize(f.name, format="xml")
        valid_file = f.name

    try:
        resolver = RDFLibOntologyResolver(
            ontology_file=["nonexistent_file_1.owl", valid_file, "nonexistent_file_2.owl"]
        )

        assert resolver.graph is not None

        assert "car" in resolver.lookup["classes"]
        assert "audi" in resolver.lookup["individuals"]

        match = resolver.find_closest_match("Audi", "individuals")
        assert match == "audi"

    finally:
        import os

        os.unlink(valid_file)


def test_multifile_ontology_all_files_missing():
    """Test loading multiple ontology files where all files are missing."""
    resolver = RDFLibOntologyResolver(
        ontology_file=["nonexistent_file_1.owl", "nonexistent_file_2.owl", "nonexistent_file_3.owl"]
    )

    assert resolver.graph is None

    assert resolver.lookup["classes"] == {}
    assert resolver.lookup["individuals"] == {}


def test_multifile_ontology_with_overlapping_entities():
    """Test loading multiple ontology files with overlapping/related entities."""
    ns = Namespace("http://example.org/automotive#")

    g1 = Graph()
    g1.add((ns.Vehicle, RDF.type, OWL.Class))
    g1.add((ns.Car, RDF.type, OWL.Class))
    g1.add((ns.Car, RDFS.subClassOf, ns.Vehicle))

    g2 = Graph()
    g2.add((ns.LuxuryCar, RDF.type, OWL.Class))
    g2.add((ns.LuxuryCar, RDFS.subClassOf, ns.Car))
    g2.add((ns.Mercedes, RDF.type, ns.LuxuryCar))
    g2.add((ns.BMW, RDF.type, ns.LuxuryCar))

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False) as f1:
        g1.serialize(f1.name, format="xml")
        file1_path = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False) as f2:
        g2.serialize(f2.name, format="xml")
        file2_path = f2.name

    try:
        resolver = RDFLibOntologyResolver(ontology_file=[file1_path, file2_path])

        assert "vehicle" in resolver.lookup["classes"]
        assert "car" in resolver.lookup["classes"]
        assert "luxurycar" in resolver.lookup["classes"]

        assert "mercedes" in resolver.lookup["individuals"]
        assert "bmw" in resolver.lookup["individuals"]

        nodes, relationships, start_node = resolver.get_subgraph("Mercedes", "individuals")

        uri_labels = {resolver._uri_to_key(n.uri) for n in nodes}
        assert "mercedes" in uri_labels
        assert "luxurycar" in uri_labels
        assert "car" in uri_labels
        assert "vehicle" in uri_labels

    finally:
        import os

        os.unlink(file1_path)
        os.unlink(file2_path)
