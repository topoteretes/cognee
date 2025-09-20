from typing import Any


class AttachedOntologyNode:
    """Lightweight wrapper to be able to parse any ontology solution and generalize cognee interface."""

    def __init__(self, uri: Any, category: str):
        self.uri = uri
        self.name = self._extract_name(uri)
        self.category = category

    @staticmethod
    def _extract_name(uri: Any) -> str:
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        return uri_str.rstrip("/").split("/")[-1]

    def __repr__(self):
        return f"AttachedOntologyNode(name={self.name}, category={self.category})"
