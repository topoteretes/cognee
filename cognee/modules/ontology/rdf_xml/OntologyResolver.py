import os
import difflib
from cognee.shared.logging_utils import get_logger
from collections import deque
from typing import List, Tuple, Dict, Optional, Any
from owlready2 import get_ontology, ClassConstruct, Ontology, Thing

from cognee.modules.ontology.exceptions import (
    OntologyInitializationError,
    FindClosestMatchError,
    GetSubgraphError,
)

logger = get_logger("OntologyAdapter")


class OntologyResolver:
    def __init__(
        self,
        ontology_file: Optional[str] = None,
        fallback_url: str = "http://example.org/empty_ontology",
    ):
        self.ontology_file = ontology_file
        try:
            if ontology_file and os.path.exists(ontology_file):
                self.ontology: Ontology = get_ontology(ontology_file).load()
                logger.info("Ontology loaded successfully from file: %s", ontology_file)
            else:
                logger.warning(
                    "Ontology file '%s' not found. Using fallback ontology at %s",
                    ontology_file,
                    fallback_url,
                )
                self.ontology = get_ontology(fallback_url)
            self.build_lookup()
        except Exception as e:
            logger.error("Failed to load ontology", exc_info=e)
            raise OntologyInitializationError() from e

    def build_lookup(self):
        try:
            self.lookup: Dict[str, Dict[str, Thing]] = {
                "classes": {
                    cls.name.lower().replace(" ", "_").strip(): cls
                    for cls in self.ontology.classes()
                },
                "individuals": {
                    ind.name.lower().replace(" ", "_").strip(): ind
                    for ind in self.ontology.individuals()
                },
            }
            logger.info(
                "Lookup built: %d classes, %d individuals",
                len(self.lookup["classes"]),
                len(self.lookup["individuals"]),
            )
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

    def get_subgraph(
        self, node_name: str, node_type: str = "individuals"
    ) -> Tuple[List[Any], List[Tuple[str, str, str]], Optional[Any]]:
        nodes_set = set()
        edges: List[Tuple[str, str, str]] = []
        visited_nodes = set()
        queue = deque()

        try:
            closest_match = self.find_closest_match(name=node_name, category=node_type)
            if not closest_match:
                logger.info("No close match found for '%s' in category '%s'", node_name, node_type)
                return list(nodes_set), edges, None

            node = self.lookup[node_type].get(closest_match)
            if node is None:
                logger.info("Node '%s' not found in lookup.", closest_match)
                return list(nodes_set), edges, None

            logger.info("%s match was found for found for '%s' node", node.name, node_name)

            queue.append(node)
            visited_nodes.add(node)
            nodes_set.add(node)

            while queue:
                current_node = queue.popleft()

                if hasattr(current_node, "is_a"):
                    for parent in current_node.is_a:
                        if isinstance(parent, ClassConstruct):
                            if hasattr(parent, "value") and hasattr(parent.value, "name"):
                                parent = parent.value
                            else:
                                continue
                        edges.append((current_node.name, "is_a", parent.name))
                        nodes_set.add(parent)
                        if parent not in visited_nodes:
                            visited_nodes.add(parent)
                            queue.append(parent)

                for prop in self.ontology.object_properties():
                    for target in prop[current_node]:
                        edges.append((current_node.name, prop.name, target.name))
                        nodes_set.add(target)
                        if target not in visited_nodes:
                            visited_nodes.add(target)
                            queue.append(target)

                    for source in prop.range:
                        if current_node in prop[source]:
                            edges.append((source.name, prop.name, current_node.name))
                            nodes_set.add(source)
                            if source not in visited_nodes:
                                visited_nodes.add(source)
                                queue.append(source)

            return list(nodes_set), edges, node
        except Exception as e:
            logger.error("Error in get_subgraph: %s", str(e))
            raise GetSubgraphError() from e
