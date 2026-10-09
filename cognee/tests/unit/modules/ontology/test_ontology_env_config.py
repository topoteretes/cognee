import inspect

from cognee.modules.ontology.get_default_ontology_resolver import get_ontology_resolver_from_env
from cognee.modules.ontology.ontology_env_config import (
    OntologyEnvConfig,
    normalize_ontology_mode,
)


def test_to_dict_matches_resolver_factory_signature():
    """to_dict() is splatted into get_ontology_resolver_from_env — every key must
    be a parameter of that factory, or env-configured cognify runs die in a TypeError."""
    factory_parameters = set(inspect.signature(get_ontology_resolver_from_env).parameters)

    config_keys = set(OntologyEnvConfig(_env_file=None).to_dict())

    assert config_keys <= factory_parameters


def test_ontology_mode_is_case_insensitive():
    config = OntologyEnvConfig(_env_file=None, ontology_mode="Strict")
    assert config.ontology_mode == "strict"

    config = OntologyEnvConfig(_env_file=None, ontology_mode=" ANNOTATE ")
    assert config.ontology_mode == "annotate"


def test_unknown_ontology_mode_falls_back_to_annotate():
    config = OntologyEnvConfig(_env_file=None, ontology_mode="bogus")
    assert config.ontology_mode == "annotate"


def test_normalize_ontology_mode_never_raises():
    assert normalize_ontology_mode("strict") == "strict"
    assert normalize_ontology_mode("Bogus") == "annotate"
    assert normalize_ontology_mode(None) == "annotate"
