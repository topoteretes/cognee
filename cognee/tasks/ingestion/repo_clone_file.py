"""A document file inside a repository clone that cognee made itself."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoCloneFile:
    """A document of a hosted repository that ``add()`` cloned for this call.

    Its path is local only because cognee put the clone there; the caller named
    a repository URL, never this path. Storage therefore takes it whatever
    ``ACCEPT_LOCAL_FILE_PATH`` says, which guards caller-supplied paths. Only
    cognee builds one (``resolve_code_repository_url``): HTTP inputs arrive as
    strings and uploads, so a request cannot name a file this way.
    """

    path: Path
