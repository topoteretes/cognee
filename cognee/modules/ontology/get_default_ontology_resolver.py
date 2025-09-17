from typing import Optional

from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy


def get_default_ontology_resolver(ontology_file: Optional[str] = None) -> BaseOntologyResolver:
    """Get the default ontology resolver (RDFLib with fuzzy matching).

    Args:
        ontology_file: Optional path to ontology file

    Returns:
        Default RDFLib ontology resolver with fuzzy matching strategy
    """
    fuzzy_strategy = FuzzyMatchingStrategy()
    return RDFLibOntologyResolver(ontology_file=ontology_file, matching_strategy=fuzzy_strategy)
