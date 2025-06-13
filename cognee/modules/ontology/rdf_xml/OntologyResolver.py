import os
import difflib
from cognee.shared.logging_utils import get_logger
from collections import deque
from typing import List, Tuple, Dict, Optional, Any
from rdflib import Graph, URIRef, RDF, RDFS, OWL

from cognee.modules.ontology.exceptions import (
    OntologyInitializationError,
    FindClosestMatchError,
    GetSubgraphError,
)

logger = get_logger("OntologyAdapter")


class AttachedOntologyNode:
    """Lightweight wrapper to be able to parse any ontology solution and generalize cognee interface."""

    def __init__(self, uri: URIRef, category: str):
        self.uri = uri
        self.name = self._extract_name(uri)
        self.category = category

    @staticmethod
    def _extract_name(uri: URIRef) -> str:
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        return uri_str.rstrip("/").split("/")[-1]

    def __repr__(self):
        return f"AttachedOntologyNode(name={self.name}, category={self.category})"


class OntologyResolver:
    def __init__(self, ontology_file: Optional[str] = None):
        self.ontology_file = ontology_file
        try:
            if ontology_file and os.path.exists(ontology_file):
                self.graph = Graph()
                self.graph.parse(ontology_file)
                logger.info("Ontology loaded successfully from file: %s", ontology_file)
            else:
                logger.info(
                    "Ontology file '%s' not found. No owl ontology will be attached to the graph.",
                    ontology_file,
                )
                self.graph = None
            self.build_lookup()
        except Exception as e:
            logger.error("Failed to load ontology", exc_info=e)
            raise OntologyInitializationError() from e

    def _uri_to_key(self, uri: URIRef) -> str:
        uri_str = str(uri)
        if "#" in uri_str:
            name = uri_str.split("#")[-1]
        else:
            name = uri_str.rstrip("/").split("/")[-1]
        return name.lower().replace(" ", "_").strip()

    def build_lookup(self):
        try:
            classes: Dict[str, URIRef] = {}
            individuals: Dict[str, URIRef] = {}

            if not self.graph:
                self.lookup: Dict[str, Dict[str, URIRef]] = {
                    "classes": classes,
                    "individuals": individuals,
                }

                return None

            for cls in self.graph.subjects(RDF.type, OWL.Class):
                key = self._uri_to_key(cls)
                classes[key] = cls

            for subj, _, obj in self.graph.triples((None, RDF.type, None)):
                if obj in classes.values():
                    key = self._uri_to_key(subj)
                    individuals[key] = subj

            self.lookup = {
                "classes": classes,
                "individuals": individuals,
            }
            logger.info(
                "Lookup built: %d classes, %d individuals",
                len(classes),
                len(individuals),
            )

            return None
        except Exception as e:
            logger.error("Failed to build lookup dictionary: %s", str(e))
            raise RuntimeError("Lookup build failed") from e

    def refresh_lookup(self):
        self.build_lookup()
        logger.info("Ontology lookup refreshed.")

    def find_closest_match(self, name: str, category: str) -> Optional[str]:
        try:
            normalized_name = name.lower().replace(" ", "_").strip()
            possible_matches = list(self.lookup.get(category, {}).keys())
            if normalized_name in possible_matches:
                return normalized_name

            best_match = difflib.get_close_matches(
                normalized_name, possible_matches, n=1, cutoff=0.8
            )
            return best_match[0] if best_match else None
        except Exception as e:
            logger.error("Error in find_closest_match: %s", str(e))
            raise FindClosestMatchError() from e

    def _get_category(self, uri: URIRef) -> str:
        if uri in self.lookup.get("classes", {}).values():
            return "classes"
        if uri in self.lookup.get("individuals", {}).values():
            return "individuals"
        return "unknown"

    def get_subgraph(
        self, node_name: str, node_type: str = "individuals", directed: bool = True
    ) -> Tuple[List[Any], List[Tuple[str, str, str]], Optional[Any]]:
        nodes_set = set()
        edges: List[Tuple[str, str, str]] = []
        visited = set()
        queue = deque()

        try:
            closest_match = self.find_closest_match(name=node_name, category=node_type)
            if not closest_match:
                logger.info("No close match found for '%s' in category '%s'", node_name, node_type)
                return [], [], None

            node = self.lookup[node_type].get(closest_match)
            if node is None:
                logger.info("Node '%s' not found in lookup.", closest_match)
                return [], [], None

            logger.info("%s match was found for found for '%s' node", node, node_name)

            queue.append(node)
            visited.add(node)
            nodes_set.add(node)

            obj_props = set(self.graph.subjects(RDF.type, OWL.ObjectProperty))

            while queue:
                current = queue.popleft()
                current_label = self._uri_to_key(current)

                if node_type == "individuals":
                    for parent in self.graph.objects(current, RDF.type):
                        parent_label = self._uri_to_key(parent)
                        edges.append((current_label, "is_a", parent_label))
                        if parent not in visited:
                            visited.add(parent)
                            queue.append(parent)
                        nodes_set.add(parent)

                for parent in self.graph.objects(current, RDFS.subClassOf):
                    parent_label = self._uri_to_key(parent)
                    edges.append((current_label, "is_a", parent_label))
                    if parent not in visited:
                        visited.add(parent)
                        queue.append(parent)
                    nodes_set.add(parent)

                for prop in obj_props:
                    prop_label = self._uri_to_key(prop)
                    for target in self.graph.objects(current, prop):
                        target_label = self._uri_to_key(target)
                        edges.append((current_label, prop_label, target_label))
                        if target not in visited:
                            visited.add(target)
                            queue.append(target)
                        nodes_set.add(target)
                    if not directed:
                        for source in self.graph.subjects(prop, current):
                            source_label = self._uri_to_key(source)
                            edges.append((source_label, prop_label, current_label))
                            if source not in visited:
                                visited.add(source)
                                queue.append(source)
                            nodes_set.add(source)

            rdf_nodes = [
                AttachedOntologyNode(uri=uri, category=self._get_category(uri))
                for uri in list(nodes_set)
            ]
            rdf_root = (
                AttachedOntologyNode(uri=node, category=self._get_category(node))
                if node is not None
                else None
            )

            return rdf_nodes, edges, rdf_root
        except Exception as e:
            logger.error("Error in get_subgraph: %s", str(e))
            raise GetSubgraphError() from e
