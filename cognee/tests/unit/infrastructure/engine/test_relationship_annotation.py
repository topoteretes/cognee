"""Tests for the Relationship() field annotation on DataPoint models."""

import asyncio
from typing import Annotated, Optional

from cognee.infrastructure.engine import DataPoint, Edge, Relationship
from cognee.infrastructure.engine.models.FieldAnnotations import _Relationship
from cognee.modules.graph.utils.get_graph_from_model import (
    _get_relationship_key,
    _get_relationship_annotation,
    _create_edge_properties,
    get_graph_from_model,
)


# ── Test models ──


class Company(DataPoint):
    name: str = ""


class Person(DataPoint):
    name: str = ""
    friends: Annotated[list["Person"], Relationship("knows")] = []
    employer: Annotated[Optional["Company"], Relationship("works_at", weight=0.9)] = None


class PlainPerson(DataPoint):
    """No Relationship annotations — backward compat."""

    name: str = ""
    friends: list["PlainPerson"] = []


class MultiRel(DataPoint):
    name: str = ""
    authored: Annotated[list["MultiRel"], Relationship("authored_by")] = []
    reviewed: Annotated[list["MultiRel"], Relationship("reviewed_by")] = []


class WithProperties(DataPoint):
    name: str = ""
    related: Annotated[
        list["WithProperties"],
        Relationship("related_to", properties={"source": "annotation"}),
    ] = []


# ── _Relationship marker tests ──


class TestRelationshipMarker:
    def test_basic_creation(self):
        r = _Relationship(label="knows")
        assert r.label == "knows"
        assert r.weight is None
        assert r.properties == {}

    def test_with_weight(self):
        r = _Relationship(label="works_at", weight=0.9)
        assert r.weight == 0.9

    def test_with_properties(self):
        r = _Relationship(label="x", properties={"key": "val"})
        assert r.properties == {"key": "val"}

    def test_repr(self):
        r = _Relationship(label="knows")
        assert "Relationship" in repr(r)
        assert "knows" in repr(r)

    def test_repr_with_weight(self):
        r = _Relationship(label="x", weight=0.5)
        assert "weight=0.5" in repr(r)


# ── Factory function tests ──


class TestRelationshipFactory:
    def test_returns_relationship_instance(self):
        r = Relationship("knows")
        assert isinstance(r, _Relationship)
        assert r.label == "knows"

    def test_default_description(self):
        r = Relationship("knows")
        assert r.description == "Defines graph relationship"


# ── Annotation lookup tests ──


class TestGetRelationshipAnnotation:
    def test_finds_annotation(self):
        rel = _get_relationship_annotation(Person, "friends")
        assert rel is not None
        assert rel.label == "knows"

    def test_finds_annotation_with_weight(self):
        rel = _get_relationship_annotation(Person, "employer")
        assert rel is not None
        assert rel.label == "works_at"
        assert rel.weight == 0.9

    def test_returns_none_for_unannotated(self):
        rel = _get_relationship_annotation(PlainPerson, "friends")
        assert rel is None

    def test_returns_none_for_nonexistent_field(self):
        rel = _get_relationship_annotation(Person, "nonexistent")
        assert rel is None


# ── _get_relationship_key tests ──


class TestGetRelationshipKey:
    def test_annotation_overrides_field_name(self):
        key = _get_relationship_key("friends", None, model_class=Person)
        assert key == "knows"

    def test_edge_metadata_overrides_annotation(self):
        edge = Edge(relationship_type="besties")
        key = _get_relationship_key("friends", edge, model_class=Person)
        assert key == "besties"

    def test_no_annotation_falls_back_to_field_name(self):
        key = _get_relationship_key("friends", None, model_class=PlainPerson)
        assert key == "friends"

    def test_no_model_class_falls_back_to_field_name(self):
        key = _get_relationship_key("friends", None)
        assert key == "friends"

    def test_empty_edge_relationship_type_uses_annotation(self):
        edge = Edge(relationship_type="")
        key = _get_relationship_key("friends", edge, model_class=Person)
        assert key == "knows"

    def test_edge_weight_only_uses_annotation_label(self):
        edge = Edge(weight=0.5)
        key = _get_relationship_key("friends", edge, model_class=Person)
        assert key == "knows"


# ── _create_edge_properties tests ──


class TestCreateEdgeProperties:
    def test_annotation_weight_applied(self):
        rel = _Relationship(label="works_at", weight=0.9)
        props = _create_edge_properties("src", "tgt", "works_at", None, rel)
        assert props["weight"] == 0.9

    def test_annotation_properties_applied(self):
        rel = _Relationship(label="x", properties={"source": "annotation"})
        props = _create_edge_properties("src", "tgt", "x", None, rel)
        assert props["source"] == "annotation"

    def test_edge_metadata_overrides_annotation_weight(self):
        rel = _Relationship(label="x", weight=0.5)
        edge = Edge(weight=0.8)
        props = _create_edge_properties("src", "tgt", "x", edge, rel)
        assert props["weight"] == 0.8

    def test_no_annotation_still_works(self):
        props = _create_edge_properties("src", "tgt", "friends", None, None)
        assert props["relationship_name"] == "friends"
        assert "weight" not in props


# ── Full graph extraction tests ──


class TestGraphExtraction:
    def test_annotation_label_in_edges(self):
        alice = Person(name="Alice")
        bob = Person(name="Bob")
        alice.friends = [bob]

        nodes, edges = asyncio.run(get_graph_from_model(alice))
        assert len(edges) >= 1
        # The edge should use "knows" not "friends"
        rel_names = [e[2] for e in edges]
        assert "knows" in rel_names
        assert "friends" not in rel_names

    def test_annotation_weight_in_edge_properties(self):
        person = Person(name="Alice")
        company = Company(name="Acme")
        person.employer = company

        nodes, edges = asyncio.run(get_graph_from_model(person))
        works_at_edges = [e for e in edges if e[2] == "works_at"]
        assert len(works_at_edges) == 1
        assert works_at_edges[0][3]["weight"] == 0.9

    def test_no_annotation_uses_field_name(self):
        a = PlainPerson(name="A")
        b = PlainPerson(name="B")
        a.friends = [b]

        nodes, edges = asyncio.run(get_graph_from_model(a))
        rel_names = [e[2] for e in edges]
        assert "friends" in rel_names

    def test_instance_edge_overrides_annotation(self):
        alice = Person(name="Alice")
        bob = Person(name="Bob")
        alice.friends = [(Edge(relationship_type="besties"), bob)]

        nodes, edges = asyncio.run(get_graph_from_model(alice))
        rel_names = [e[2] for e in edges]
        assert "besties" in rel_names
        assert "knows" not in rel_names

    def test_multiple_relationship_fields(self):
        a = MultiRel(name="A")
        b = MultiRel(name="B")
        c = MultiRel(name="C")
        a.authored = [b]
        a.reviewed = [c]

        nodes, edges = asyncio.run(get_graph_from_model(a))
        rel_names = {e[2] for e in edges}
        assert "authored_by" in rel_names
        assert "reviewed_by" in rel_names
        assert "authored" not in rel_names
        assert "reviewed" not in rel_names

    def test_annotation_properties_in_edge_data(self):
        a = WithProperties(name="A")
        b = WithProperties(name="B")
        a.related = [b]

        nodes, edges = asyncio.run(get_graph_from_model(a))
        related_edges = [e for e in edges if e[2] == "related_to"]
        assert len(related_edges) == 1
        assert related_edges[0][3]["source"] == "annotation"
