from typing import Optional

from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    ensure_ontology_usable_in_strict_mode,
)
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.ontology_env_config import (
    get_ontology_env_config,
    normalize_ontology_mode,
)
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy


def get_default_ontology_resolver() -> BaseOntologyResolver:
    return RDFLibOntologyResolver(ontology_file=None, matching_strategy=FuzzyMatchingStrategy())


def get_configured_ontology_resolver(
    config: Optional[Config] = None,
) -> Optional[BaseOntologyResolver]:
    """Resolve the ontology resolver from an explicit config or the environment."""
    if config is not None:
        ontology_config = config.get("ontology_config")
        if isinstance(ontology_config, dict) and "ontology_resolver" in ontology_config:
            return ontology_config["ontology_resolver"]
        return None

    ontology_config = get_ontology_env_config()
    if (
        ontology_config.ontology_file_path
        and ontology_config.ontology_resolver
        and ontology_config.matching_strategy
    ):
        resolver = get_ontology_resolver_from_env(**ontology_config.to_dict())
        if ontology_config.ontology_mode == "strict":
            # Fail before any pipeline work: a mistyped ONTOLOGY_FILE_PATH yields an
            # empty resolver, and strict mode over an empty ontology drops everything.
            ensure_ontology_usable_in_strict_mode(resolver)
        return resolver
    return None


def get_configured_ontology_mode(config: Optional[Config] = None) -> str:
    """Resolve the ontology mode from an explicit config or the environment.

    A per-call ``ontology_mode`` in the config wins; otherwise the ONTOLOGY_MODE
    environment value applies. The result is always a normalized, valid mode.
    """
    if config is not None:
        ontology_config = config.get("ontology_config")
        if isinstance(ontology_config, dict) and ontology_config.get("ontology_mode") is not None:
            return normalize_ontology_mode(ontology_config["ontology_mode"])

    return get_ontology_env_config().ontology_mode


def get_ontology_resolver_from_env(
    ontology_resolver: str = "", matching_strategy: str = "", ontology_file_path: str = ""
) -> BaseOntologyResolver:
    """
    Create and return an ontology resolver instance based on environment parameters.

    Currently, this function supports only the RDFLib-based ontology resolver
    with a fuzzy matching strategy.

    Args:
        ontology_resolver (str): The ontology resolver type to use.
            Supported value: "rdflib".
        matching_strategy (str): The matching strategy to apply.
            Supported value: "fuzzy".
        ontology_file_path (str): Path to the ontology file(s) required for the resolver.
            Can be a single path or comma-separated paths for multiple files.

    Returns:
        BaseOntologyResolver: An instance of the requested ontology resolver.

    Raises:
        EnvironmentError: If the provided resolver or strategy is unsupported,
            or if required parameters are missing.
    """
    if ontology_resolver == "rdflib" and matching_strategy == "fuzzy" and ontology_file_path:
        if "," in ontology_file_path:
            file_paths = [path.strip() for path in ontology_file_path.split(",")]
        else:
            file_paths = ontology_file_path

        return RDFLibOntologyResolver(
            matching_strategy=FuzzyMatchingStrategy(), ontology_file=file_paths
        )
    else:
        raise EnvironmentError(
            f"Unsupported ontology resolver: {ontology_resolver}. "
            f"Supported resolvers are: RdfLib with FuzzyMatchingStrategy."
        )
