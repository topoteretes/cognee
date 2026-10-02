"""Per-call options for the code graph routes (CODE and CODE_REPO)."""

from typing import TypedDict, cast


class CodeGraphConfig(TypedDict, total=False):
    """Options for code files and code repositories, passed as ``codegraph_config``.

    ``add()`` reads ``repo_credentials`` and ``include_documents``; ``cognify()``
    reads ``index_vectors``; ``remember()`` hands the same dict to both, so each
    takes what it needs.
    """

    # Also embed the code facts the run builds, so completion search types can
    # reach the code. Off by default: SearchType.CODE reads the graph only.
    index_vectors: bool
    # Also ingest the repository's document files (README, docs, ...) as
    # ordinary documents, alongside its code graph. On by default, as a
    # directory add has always done. False indexes the code graph only -- the
    # behaviour the removed ``content_type="code"`` route had, and what the
    # GitHub sync wants: no LLM extraction or embeddings for a repo's prose.
    # Documents are excluded regardless when no LLM API key is configured,
    # since their pipelines need one.
    include_documents: bool
    # Token for cloning private GitHub/GitLab repository URLs (e.g. a GitHub App
    # installation token). It reaches git only through environment config,
    # never the URL, so nothing stored or logged carries it.
    repo_credentials: str


CODEGRAPH_CONFIG_KEYS = frozenset(CodeGraphConfig.__annotations__)


def validate_codegraph_config(codegraph_config: CodeGraphConfig | None) -> CodeGraphConfig:
    """Return the config as a dict, rejecting keys the code graph routes do not read."""
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
    return cast(CodeGraphConfig, dict(codegraph_config))
