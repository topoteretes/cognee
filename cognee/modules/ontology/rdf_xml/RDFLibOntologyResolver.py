import os
import difflib
from cognee.shared.logging_utils import get_logger
from collections import deque
from typing import List, Tuple, Dict, Optional, Any, Union, IO
from rdflib import Graph, URIRef, RDF, RDFS, OWL
from rdflib.util import guess_format

from cognee.modules.ontology.exceptions import (
    OntologyInitializationError,
    FindClosestMatchError,
    GetSubgraphError,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.ontology.matching_strategies import MatchingStrategy, FuzzyMatchingStrategy

logger = get_logger("OntologyAdapter")

CONTENT_TYPE_FORMATS = {
    "application/rdf+xml": "xml",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/turtle": "turtle",
    "application/x-turtle": "turtle",
    "text/n3": "n3",
    "application/n-triples": "nt",
    "application/n-quads": "nquads",
    "application/trig": "trig",
    "application/ld+json": "json-ld",
}

FALLBACK_FORMATS = ("xml", "turtle", "n3", "nt", "json-ld", "trig", "nquads")


class RDFLibOntologyResolver(BaseOntologyResolver):
    """RDFLib-based ontology resolver implementation.

    This implementation uses RDFLib to parse and work with RDF/OWL ontology files.
    It provides fuzzy matching and subgraph extraction capabilities for ontology entities.
    """

    def __init__(
        self,
        ontology_file: Optional[Union[str, List[str], IO, List[IO]]] = None,
        matching_strategy: Optional[MatchingStrategy] = None,
    ) -> None:
        super().__init__(matching_strategy)
        self.ontology_file = ontology_file
        try:
            self.graph = None
            if ontology_file is not None:
                files_to_load = []
                file_objects = []

                if hasattr(ontology_file, "read"):
                    file_objects = [ontology_file]
                elif isinstance(ontology_file, str):
                    files_to_load = [ontology_file]
                elif isinstance(ontology_file, list):
                    if all(hasattr(item, "read") for item in ontology_file):
                        file_objects = ontology_file
                    else:
                        files_to_load = ontology_file
                else:
                    raise ValueError(
                        f"ontology_file must be a string, list of strings, file-like object, list of file-like objects, or None. Got: {type(ontology_file)}"
                    )

                if file_objects:
                    self.graph = Graph()
                    loaded_objects = []
                    for file_obj in file_objects:
                        try:
                            parsed_format = self._parse_file_object(file_obj, self.graph)
                            loaded_objects.append(file_obj)
                            logger.info(
                                "Ontology loaded successfully from file object '%s' as %s",
                                self._get_file_object_name(file_obj),
                                parsed_format,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to parse ontology file object '%s': %s",
                                self._get_file_object_name(file_obj),
                                str(e),
                            )

                    if not loaded_objects:
                        raise ValueError(
                            "No valid ontology file objects could be parsed. "
                            "No owl ontology will be attached to the graph."
                        )
                    else:
                        logger.info("Total ontology file objects loaded: %d", len(loaded_objects))

                elif files_to_load:
                    self.graph = Graph()
                    loaded_files = []
                    for file_path in files_to_load:
                        if os.path.exists(file_path):
                            self.graph.parse(file_path)
                            loaded_files.append(file_path)
                            logger.info("Ontology loaded successfully from file: %s", file_path)
                        else:
                            logger.warning(
                                "Ontology file '%s' not found. Skipping this file.",
                                file_path,
                            )

                    if not loaded_files:
                        logger.info(
                            "No valid ontology files found. No owl ontology will be attached to the graph."
                        )
                        self.graph = None
                    else:
                        logger.info("Total ontology files loaded: %d", len(loaded_files))
                else:
                    logger.info(
                        "No ontology file provided. No owl ontology will be attached to the graph."
                    )
            else:
                logger.info(
                    "No ontology file provided. No owl ontology will be attached to the graph."
                )
                self.graph = None

            self.build_lookup()
        except Exception as e:
            logger.error("Failed to load ontology", exc_info=True)
            raise OntologyInitializationError(f"Failed to load ontology: {e}") from e

    def _uri_to_key(self, uri: URIRef) -> str:
        uri_str = str(uri)
        if "#" in uri_str:
            name = uri_str.split("#")[-1]
        else:
            name = uri_str.rstrip("/").split("/")[-1]
        return name.lower().replace(" ", "_").strip()

    def _get_file_object_name(self, file_obj: IO) -> str:
        return str(
            getattr(file_obj, "filename", None)
            or getattr(file_obj, "name", None)
            or file_obj.__class__.__name__
        )

    def _get_content_type_format(self, file_obj: IO) -> Optional[str]:
        content_type = getattr(file_obj, "content_type", None)
        if not content_type:
            return None

        content_type = str(content_type).split(";", maxsplit=1)[0].strip().lower()
        return CONTENT_TYPE_FORMATS.get(content_type)

    def _get_candidate_formats(self, file_obj: IO) -> List[str]:
        formats = []

        filename = getattr(file_obj, "filename", None) or getattr(file_obj, "name", None)
        if filename:
            guessed_format = guess_format(str(filename))
            if guessed_format:
                formats.append(guessed_format)

        content_type_format = self._get_content_type_format(file_obj)
        if content_type_format:
            formats.append(content_type_format)

        formats.extend(FALLBACK_FORMATS)
        return list(dict.fromkeys(formats))

    def _parse_file_object(self, file_obj: IO, target_graph: Graph) -> str:
        try:
            file_obj.seek(0)
        except (AttributeError, OSError):
            pass

        content = file_obj.read()
        if not isinstance(content, (str, bytes)):
            raise TypeError(
                f"Ontology file object returned unsupported content type: {type(content)}"
            )

        candidate_formats = self._get_candidate_formats(file_obj)
        parse_errors = []

        for rdf_format in candidate_formats:
            parsed_graph = Graph()
            try:
                parsed_graph.parse(data=content, format=rdf_format)
            except Exception as error:
                parse_errors.append(f"{rdf_format}: {error}")
                continue

            for prefix, namespace in parsed_graph.namespaces():
                target_graph.bind(prefix, namespace, override=False)

            for triple in parsed_graph:
                target_graph.add(triple)

            return rdf_format

        raise ValueError(
            f"Unable to parse ontology file object '{self._get_file_object_name(file_obj)}'. "
            f"Tried formats: {', '.join(candidate_formats)}. "
            f"Last error: {parse_errors[-1] if parse_errors else 'unknown error'}"
        )

    def build_lookup(self) -> None:
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

    def refresh_lookup(self) -> None:
        self.build_lookup()
        logger.info("Ontology lookup refreshed.")

    def find_closest_match(self, name: str, category: str) -> Optional[str]:
        try:
            normalized_name = name.lower().replace(" ", "_").strip()
            possible_matches = list(self.lookup.get(category, {}).keys())

            return self.matching_strategy.find_match(normalized_name, possible_matches)
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
    ) -> Tuple[
        List[AttachedOntologyNode], List[Tuple[str, str, str]], Optional[AttachedOntologyNode]
    ]:
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
