import os
import difflib
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional
from owlready2 import get_ontology, ClassConstruct, Ontology, Thing

logger = logging.getLogger("OntologyAdapter")


class OntologyAdapter:
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
            self._build_lookup()
        except Exception as e:
            logger.error("Failed to load ontology: %s", str(e))
            raise RuntimeError("Ontology initialization failed") from e

    def _build_lookup(self):
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
        self._build_lookup()
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
            raise e

    def get_subgraph(
        self, node_name: str, node_type: str = "individuals"
    ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        nodes = set()
        relationships: List[Tuple[str, str, str]] = []
        visited_nodes = set()
        queue = deque()

        try:
            closest_match = self.find_closest_match(name=node_name, category=node_type)
            if not closest_match:
                logger.info("No close match found for '%s' in category '%s'", node_name, node_type)
                return list(nodes), relationships, None

            node = self.lookup[node_type].get(closest_match)
            if node is None:
                logger.info("Node '%s' not found in lookup.", closest_match)
                return list(nodes), relationships, None

            logger.info("%s match was found for found for '%s' node", node.name, node_name)

            queue.append(node)
            visited_nodes.add(node)
            nodes.add(node)

            while queue:
                current_node = queue.popleft()

                if hasattr(current_node, "is_a"):
                    for parent in current_node.is_a:
                        if isinstance(parent, ClassConstruct):
                            if hasattr(parent, "value") and hasattr(parent.value, "name"):
                                parent = parent.value
                            else:
                                continue
                        relationships.append((current_node.name, "is_a", parent.name))
                        nodes.add(parent)
                        if parent not in visited_nodes:
                            visited_nodes.add(parent)
                            queue.append(parent)

                for prop in self.ontology.object_properties():
                    for target in prop[current_node]:
                        relationships.append((current_node.name, prop.name, target.name))
                        nodes.add(target)
                        if target not in visited_nodes:
                            visited_nodes.add(target)
                            queue.append(target)

                    for source in prop.range:
                        if current_node in prop[source]:
                            relationships.append((source.name, prop.name, current_node.name))
                            nodes.add(source)
                            if source not in visited_nodes:
                                visited_nodes.add(source)
                                queue.append(source)

            return list(nodes), relationships, node
        except Exception as e:
            logger.error("Error in get_subgraph: %s", str(e))
            raise RuntimeError("Failed to retrieve subgraph") from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        adapter = OntologyAdapter(ontology_file="basic_ontology.owl")

        nodes, relationships = adapter.get_subgraph("Audi", node_type="individuals")
        logger.info("Subgraph nodes: %s", nodes)
        logger.info("Subgraph relationships: %s", relationships)

    except Exception as e:
        logger.error("Ontology adapter error: %s", e)
