"""Per-call options for the code graph routes (CODE and CODE_REPO)."""

from typing import TypedDict, cast


class CodeGraphConfig(TypedDict, total=False):
    """Options for code files and code repositories, passed as ``codegraph_config``.

    ``add()`` reads ``repo_credentials``, ``include_documents`` and
    ``treat_as_repository``; ``cognify()`` reads ``index_vectors``;
    ``remember()`` hands the same dict to both, so each takes what it needs.
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
    # Every item in this call's data IS a repository spec -- a local directory
    # or a git remote -- rather than something to sniff. Off by default: add()
    # recognises repositories on its own (``code_repo_clone_url`` for URLs,
    # ``detect_code_project`` for directories), which is deliberately
    # conservative because most http(s) URLs really are web pages and most
    # directories really are document trees.
    #
    # Turn it on when that sniffing cannot see what you have: a Bitbucket or
    # self-hosted forge URL with no ``.git`` suffix (guessing would scrape a web
    # page instead), an ssh remote (``git@host:owner/repo``), or a source tree
    # with no build manifest and no ``.git``. It is also what the deprecated
    # ``content_type="code"`` declared, and what that value is read as, so those
    # callers keep the repository set they had.
    #
    # Every item is then resolved through ``resolve_repo_source``: a local
    # directory is used in place, a remote URL is shallow-cloned. An item that
    # is neither raises instead of being stored as text.
    treat_as_repository: bool


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
