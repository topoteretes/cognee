"""What git leaves in a depth-1 clone, for tests that fake git or build a clone by hand.

A clone made without ``--branch`` has ``refs/remotes/origin/HEAD``; one made with it does
not, and its HEAD names the branch (checked on git 2.50.1). resolve_repo asks git for
exactly this to tell the clone of the default branch from the clone of a chosen one.
"""

from pathlib import Path


def write_clone_state(clone_dir: Path, branch: str | None = None) -> Path:
    """Make ``clone_dir`` look like a clone of ``branch`` (None: of the default branch)."""
    git_dir = clone_dir / ".git"
    # HEAD, objects and refs are what git needs to take a directory for a repository.
    (git_dir / "objects").mkdir(parents=True, exist_ok=True)
    (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    if branch is None:
        origin = git_dir / "refs" / "remotes" / "origin"
        origin.mkdir(parents=True, exist_ok=True)
        (origin / "HEAD").write_text("ref: refs/remotes/origin/main\n")
        branch = "main"
    (git_dir / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    return clone_dir


def write_cloned_state(clone_args: list[str]) -> None:
    """The state a ``git clone`` with these arguments leaves in its destination, the last one."""
    branch = clone_args[clone_args.index("--branch") + 1] if "--branch" in clone_args else None
    write_clone_state(Path(clone_args[-1]), branch)
