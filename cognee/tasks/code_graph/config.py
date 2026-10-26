"""Per-call options for the code graph routes, passed as ``codegraph_config``.

The keys and their meaning are those of ``codegraph_config`` in the stacked
SDK-793 work, so that work extends this type instead of replacing it.
"""

from typing import TypedDict, cast


class CodeGraphConfig(TypedDict, total=False):
    """Options for code repositories, read by ``add()``."""

    # Also ingest the repository's document files (README, docs, ...) as
    # ordinary documents, alongside its code graph. On by default, as a
    # directory add has always done. False indexes the code graph only -- what
    # the GitHub sync wants: no LLM extraction or embeddings for a repo's prose.
    include_documents: bool
    # Token for cloning private GitHub/GitLab repository URLs (e.g. a GitHub App
    # installation token). It reaches git only through environment config,
    # never the URL, so nothing stored or logged carries it.
    repo_credentials: str


CODEGRAPH_CONFIG_KEYS = frozenset(CodeGraphConfig.__annotations__)


def validate_codegraph_config(codegraph_config: CodeGraphConfig | None) -> CodeGraphConfig:
    """Return the config as a dict, rejecting unknown keys and mistyped values.

    A key set to ``None`` counts as absent and is dropped. ``include_documents`` must be
    a real bool: ``"false"`` is truthy, so accepting strings would invert it.
    """
    if codegraph_config is None:
        return {}
    if not isinstance(codegraph_config, dict):
        raise TypeError(f"codegraph_config must be a dict, got {type(codegraph_config).__name__}.")
    unknown = set(codegraph_config) - CODEGRAPH_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown codegraph_config keys: {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(CODEGRAPH_CONFIG_KEYS))}."
        )
    config = {key: value for key, value in codegraph_config.items() if value is not None}
    if "include_documents" in config and not isinstance(config["include_documents"], bool):
        raise TypeError(
            "codegraph_config['include_documents'] must be a bool, "
            f"got {type(config['include_documents']).__name__}."
        )
    credentials = config.get("repo_credentials")
    if credentials is not None and (not isinstance(credentials, str) or not credentials):
        raise TypeError("codegraph_config['repo_credentials'] must be a non-empty str.")
    return cast(CodeGraphConfig, config)
