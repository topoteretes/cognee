#!/usr/bin/env python3
"""
Check that the docstring pilot changed docstrings, and nothing else.

The value of a docstring-only pull request is that a maintainer can read it as
prose. One stray code edit costs that: the diff now has to be reviewed as a code
change, and the reviewer has to find the edit before they can trust the rest.
The skill instructs the agent to stay inside docstrings; this enforces it.

How the check works:

1. Exactly one file may differ from the base commit, and it must be the file the
   run was dispatched for. An edit to a second file fails here even if that edit
   is itself a docstring.
2. For that file, the base and working versions are parsed, every docstring is
   removed from both trees, and the results are compared with ``ast.dump``.
   Removing rather than blanking is what lets a docstring be *added* where there
   was none: both sides normalize to the same body. A body left empty by the
   removal gets a ``pass`` so the trees stay comparable.
3. Comments are compared separately, as a multiset of ``tokenize`` COMMENT
   values. Comments are absent from the AST, so without this step the agent
   could rewrite them freely.

Known blind spot: pure whitespace and formatting changes outside docstrings are
invisible to both comparisons. The workflow runs ``ruff format`` on the file
anyway, so such a change would be normalized rather than reviewed — noise in the
diff at worst, never a behavior change.

Exit codes: 0 = docstring-only (or no change at all), 1 = other changes found,
2 = a version of the file could not be parsed.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import subprocess
import sys
import tokenize
from collections import Counter

DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def git(*command: str, cwd: str) -> str:
    result = subprocess.run(["git", *command], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove every docstring in ``tree``, keeping each body syntactically valid."""
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_shape(source: str, label: str) -> str:
    """A canonical string for everything in ``source`` except its docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        print(f"::error::Could not parse the {label} version of the file: {error}")
        raise SystemExit(2)
    return ast.dump(strip_docstrings(tree))


def comments(source: str, label: str) -> Counter:
    """Multiset of comment texts, since comments never reach the AST."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return Counter(tok.string.strip() for tok in tokens if tok.type == tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        print(f"::error::Could not tokenize the {label} version of the file: {error}")
        raise SystemExit(2)


def fail(reason: str, detail: str = "") -> int:
    lines = ["### Docstring pilot — diff rejected", "", reason]
    if detail:
        lines += ["", "```", detail.rstrip(), "```"]
    message = "\n".join(lines)
    print(message)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(message + "\n")
    print(f"::error::{reason}")
    return 1


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="The file the run was dispatched for")
    parser.add_argument("--base", default="HEAD", help="Commit the working tree is compared to")
    parser.add_argument("--repo-root", default=".", help="Repository the diff is taken in")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Path prefix to exclude from the untracked-file check, repeatable. The "
            "workflow checks out the pilot machinery into a subdirectory of the tree "
            "being edited, which is untracked there but not something the agent wrote."
        ),
    )
    args = parser.parse_args()

    root = args.repo_root
    changed = [
        line for line in git("diff", "--name-only", args.base, cwd=root).splitlines() if line
    ]
    untracked = [
        line
        for line in git("ls-files", "--others", "--exclude-standard", cwd=root).splitlines()
        if line
    ]

    # The agent is asked to write a ledger file into the workspace, so an
    # untracked markdown file is expected. An untracked *source* file is not.
    stray = [
        path
        for path in untracked
        if not path.endswith(".md") and not path.startswith(tuple(args.ignore) or ("\0",))
    ]
    if stray:
        return fail(
            "The run created files outside the target. Only docstring edits to the target "
            "file and the markdown ledger are allowed.",
            "\n".join(stray),
        )

    if not changed:
        print(f"No changes to `{args.path}` — nothing to check.")
        write_output("changed", "false")
        return 0

    if changed != [args.path]:
        return fail(
            f"The run was dispatched for `{args.path}` but modified {len(changed)} file(s). "
            "A pilot run must touch exactly the file it was given.",
            "\n".join(changed),
        )

    base_source = git("show", f"{args.base}:{args.path}", cwd=root)
    with open(os.path.join(root, args.path), encoding="utf-8") as handle:
        new_source = handle.read()

    if code_shape(base_source, "base") != code_shape(new_source, "edited"):
        return fail(
            f"`{args.path}` has changes outside its docstrings. Comparing the two versions "
            "with all docstrings removed shows a different syntax tree, so code, imports, "
            "annotations, or defaults were altered.",
            git("diff", args.base, "--", args.path, cwd=root),
        )

    base_comments = comments(base_source, "base")
    new_comments = comments(new_source, "edited")
    if base_comments != new_comments:
        added = new_comments - base_comments
        removed = base_comments - new_comments
        detail = "\n".join(
            [*(f"+ {c}" for c in sorted(added)), *(f"- {c}" for c in sorted(removed))]
        )
        return fail(
            f"`{args.path}` has comment changes. Comments are not docstrings; the pilot's "
            "diff must be readable as prose changes to docstrings only.",
            detail,
        )

    print(f"`{args.path}` changed docstrings only.")
    write_output("changed", "true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
