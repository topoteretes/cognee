import builtins
import io
from types import SimpleNamespace

import pytest

from cognee.modules.ontology.exceptions import OntologyReasoningDependencyError
from cognee.modules.ontology.rdf_xml import RDFLibOntologyResolver as resolver_module
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

pytest.importorskip("owlrl")

# Person and Human are equivalent classes; worksFor is typed by domain/range and has
# an inverse; bob is never given an rdf:type, only a worksFor edge.
ONTOLOGY = """
@prefix ex: <http://example.org/test#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Person a owl:Class .
ex:Human a owl:Class ; owl:equivalentClass ex:Person .
ex:Company a owl:Class .

ex:worksFor a owl:ObjectProperty ; rdfs:domain ex:Person ; rdfs:range ex:Company .
ex:employs a owl:ObjectProperty ; owl:inverseOf ex:worksFor .

ex:alice a ex:Person .
ex:acme a ex:Company .
ex:bob ex:worksFor ex:acme .
"""


def _resolver(reasoning: str) -> RDFLibOntologyResolver:
    return RDFLibOntologyResolver(ontology_file=io.StringIO(ONTOLOGY), ontology_reasoning=reasoning)


def test_none_reasoning_uses_asserted_triples_only():
    resolver = _resolver("none")

    _, alice_edges, _ = resolver.get_subgraph("alice")
    assert ("alice", "is_a", "person") in alice_edges
    # Human is reachable only through owl:equivalentClass, which the raw walk ignores.
    assert ("alice", "is_a", "human") not in alice_edges

    _, acme_edges, _ = resolver.get_subgraph("acme")
    assert ("acme", "employs", "bob") not in acme_edges

    # bob is typed only by rdfs:domain inference, so without reasoning he is unknown.
    assert "bob" not in resolver.lookup["individuals"]
    assert resolver.find_closest_match("bob", "individuals") is None


def test_owl_rl_reasoning_resolves_equivalent_class_inverse_and_domain_typing():
    resolver = _resolver("owl_rl")

    _, alice_edges, _ = resolver.get_subgraph("alice")
    assert ("alice", "is_a", "person") in alice_edges
    assert ("alice", "is_a", "human") in alice_edges

    _, acme_edges, _ = resolver.get_subgraph("acme")
    assert ("acme", "employs", "bob") in acme_edges

    assert "bob" in resolver.lookup["individuals"]
    _, bob_edges, _ = resolver.get_subgraph("bob")
    assert ("bob", "is_a", "person") in bob_edges
    assert ("bob", "worksfor", "acme") in bob_edges


def test_owl_rl_reasoning_keeps_closure_bookkeeping_out_of_the_graph():
    resolver = _resolver("owl_rl")

    # owl:Thing / owl:Nothing must not become matchable classes.
    assert "thing" not in resolver.lookup["classes"]
    assert "nothing" not in resolver.lookup["classes"]
    assert set(resolver.lookup["classes"]) == {"person", "human", "company"}

    nodes, edges, _ = resolver.get_subgraph("alice")
    assert all(source != target for source, relation, target in edges if relation == "is_a")
    assert "thing" not in {resolver._uri_to_key(node.uri) for node in nodes}


def test_reasoning_value_is_normalized_with_fallback():
    resolver = RDFLibOntologyResolver(
        ontology_file=io.StringIO(ONTOLOGY), ontology_reasoning=" OWL_RL "
    )
    assert resolver.ontology_reasoning == "owl_rl"
    assert "bob" in resolver.lookup["individuals"]

    resolver = RDFLibOntologyResolver(
        ontology_file=io.StringIO(ONTOLOGY), ontology_reasoning="bogus"
    )
    assert resolver.ontology_reasoning == "none"
    assert "bob" not in resolver.lookup["individuals"]


def test_reasoning_defaults_to_env_value(monkeypatch):
    monkeypatch.setattr(
        resolver_module,
        "get_ontology_env_config",
        lambda: SimpleNamespace(ontology_reasoning="owl_rl"),
    )
    resolver = RDFLibOntologyResolver(ontology_file=io.StringIO(ONTOLOGY))
    assert resolver.ontology_reasoning == "owl_rl"
    assert "bob" in resolver.lookup["individuals"]


def test_owl_rl_without_owlrl_installed_raises_configuration_error(monkeypatch):
    real_import = builtins.__import__

    def fail_owlrl_import(name, *args, **kwargs):
        if name == "owlrl":
            raise ImportError("No module named 'owlrl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_owlrl_import)

    with pytest.raises(OntologyReasoningDependencyError, match="pip install owlrl"):
        _resolver("owl_rl")


def test_none_reasoning_does_not_need_owlrl(monkeypatch):
    real_import = builtins.__import__

    def fail_owlrl_import(name, *args, **kwargs):
        if name == "owlrl":
            raise ImportError("No module named 'owlrl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_owlrl_import)

    resolver = _resolver("none")
    assert "alice" in resolver.lookup["individuals"]
