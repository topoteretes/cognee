"""Per-call options for code repositories, passed as ``codegraph_config``."""

from typing import TypedDict, cast


class CodeGraphConfig(TypedDict, total=False):
    """Options for code repositories, accepted by ``add()``, ``cognify()`` and ``remember()``.

    ``add()`` reads ``include_documents``, ``repo_credentials`` and
    ``treat_as_repository``; ``cognify()`` reads ``index_vectors``;
    ``remember()`` hands the same dict to both, so each takes what it needs.
    """

    # Also embed the code facts cognify builds for a repository, so completion
    # search types can reach the code. Off by default: SearchType.CODE reads
    # the graph only.
    index_vectors: bool
    # Also ingest the repository's document files (README, docs, ...) as
    # ordinary documents next to its code graph. On by default. False indexes
    # the code graph only. Documents are skipped regardless when no LLM API key
    # is configured, since their pipelines need one.
    include_documents: bool
    # Token for cloning private https repositories (e.g. a GitHub App
    # installation token). It reaches git only through environment config,
    # never the URL, so nothing stored or logged carries it. Not usable with
    # ssh remotes, which authenticate with the host's ssh keys.
    repo_credentials: str
    # Every item in the call is a repository: a local directory or a git
    # remote. add() otherwise recognises repositories on its own -- GitHub and
    # GitLab URLs, ``.git`` URLs, ssh remotes, URLs a git server answers for,
    # and directories with a project marker -- and that cannot see a source
    # tree with no build file and no ``.git``. An item that is neither a
    # directory nor a remote raises instead of being ingested as text.
    treat_as_repository: bool


CODEGRAPH_CONFIG_KEYS = frozenset(CodeGraphConfig.__annotations__)
_BOOL_KEYS = ("index_vectors", "include_documents", "treat_as_repository")


def validate_codegraph_config(codegraph_config: CodeGraphConfig | None) -> CodeGraphConfig:
    """Return the config as a new dict, rejecting unknown keys and mistyped values.

    A key set to ``None`` counts as absent and is dropped. The flags must be
    real bools: ``"false"`` is truthy, so accepting strings would invert them.
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
    for key in _BOOL_KEYS:
        if key in config and not isinstance(config[key], bool):
            raise TypeError(
                f"codegraph_config[{key!r}] must be a bool, got {type(config[key]).__name__}."
            )
    credentials = config.get("repo_credentials")
    if credentials is not None and (not isinstance(credentials, str) or not credentials):
        raise TypeError("codegraph_config['repo_credentials'] must be a non-empty str.")
    return cast(CodeGraphConfig, config)
