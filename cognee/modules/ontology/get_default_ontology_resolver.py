from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy


def get_default_ontology_resolver() -> RDFLibOntologyResolver:
    return RDFLibOntologyResolver(ontology_file=None, matching_strategy=FuzzyMatchingStrategy())
