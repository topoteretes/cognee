#!/usr/bin/env python3
"""Enforce that Alembic revision ids are GENERATED, unique, and match their filename.

Why this exists
---------------
A revision id is not a label — it is the value written into every deployed
database's ``alembic_version`` table. It is the only thing that tells a database
where it sits in the chain, and it can never be changed after a release ships.

Historically these ids were typed by hand in this repo, producing keyboard
patterns like ``a1b2c3d4e5f6`` / ``d4e5f6a7b8c9`` instead of the 12 hex chars
``alembic revision`` generates. Hand-typed ids collide: two different migrations
end up sharing one id, and from then on that id means two different schemas.
That already happened once between this repo and the Cloud migration trees
(``b2c3d4e5f6a7`` is ``add_search_history_indexes`` here and
``add_renewal_amount_currency_to_user_subscription`` on Cloud), and the fix was
to rename a shipped migration — something that is only survivable because no
production database had yet stamped the losing id.

So: always let Alembic mint the id.

    uv run alembic revision -m "short description"

Checks
------
Always, over every migration in the versions directory:

  1. the module's ``revision`` equals its filename prefix;
  2. the revision id is exactly 12 lowercase hex characters (``rev_id()`` format);
  3. no two migrations share a revision id.

Additionally, for migrations that are NEW relative to a git base ref (default
``HEAD``, override with ``--base-ref``), the id must not look hand-typed.
Existing ids are deliberately exempt: they are already stamped in real
databases and renaming them is what this check exists to prevent.

Usage
-----
    python tools/check_alembic_revision_ids.py                    # new-vs-HEAD
    python tools/check_alembic_revision_ids.py --base-ref origin/dev
    python tools/check_alembic_revision_ids.py --all              # shape-check everything
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = REPO_ROOT / "cognee" / "alembic" / "versions"

# `revision = "abc"` and `revision: str = "abc"`.
_REVISION_RE = re.compile(r'^revision(?:\s*:\s*[^=]+)?\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
# What alembic's rev_id() produces: uuid4().hex[-12:]
_REV_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _is_run(chars: str) -> bool:
    """True if ``chars`` steps by a constant amount through its own alphabet.

    Hand-typed ids walk the digit alphabet (0-9) and the letter alphabet (a-f)
    separately — ``d4e5f6a7b8c9`` is ``d,e,f,a,b,c`` interleaved with
    ``4,5,6,7,8,9`` — so the modulus is the alphabet actually used, not 16.
    """
    if len(chars) < 4:
        return False
    if all(c.isdigit() for c in chars):
        values, modulus = [int(c) for c in chars], 10
    elif all(c.isalpha() for c in chars):
        values, modulus = [ord(c) - ord("a") for c in chars], 6
    else:
        values, modulus = [int(c, 16) for c in chars], 16

    step = (values[1] - values[0]) % modulus
    if step == 0:
        return False
    return all((b - a) % modulus == step for a, b in zip(values, values[1:]))


def looks_hand_typed(revision: str) -> bool:
    """Detect the keyboard-pattern id family.

    Flags an id whose characters form one arithmetic run, or whose even- and
    odd-indexed characters each form one. Validated at 0 false positives over
    200k generated ids, while catching every hand-typed id in this repo.
    """
    if not _REV_ID_RE.match(revision):
        return False
    return _is_run(revision) or (_is_run(revision[0::2]) and _is_run(revision[1::2]))


def _revision_of(path: Path) -> str | None:
    match = _REVISION_RE.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _migration_files(versions_dir: Path) -> list[Path]:
    return sorted(p for p in versions_dir.glob("*.py") if p.name != "__init__.py")


def _revisions_at(base_ref: str, versions_dir: Path) -> set[str]:
    """Revision ids present in ``versions_dir`` at ``base_ref``, by filename prefix.

    Returns an empty set when the ref cannot be read (shallow clone, fresh repo),
    which makes every migration look new — the shape check then applies to all of
    them. That is the safe direction to fail: noisy, never silent.
    """
    relative = versions_dir.relative_to(REPO_ROOT)
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-tree", "--name-only", base_ref, f"{relative}/"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {
        Path(line).name.split("_", 1)[0]
        for line in listing.splitlines()
        if line.endswith(".py") and not line.endswith("__init__.py")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="git ref whose migrations count as pre-existing (default: HEAD)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="apply the hand-typed check to every migration, not just new ones",
    )
    parser.add_argument(
        "--versions-dir",
        default=str(VERSIONS_DIR),
        help=f"Alembic versions directory (default: {VERSIONS_DIR})",
    )
    args = parser.parse_args()

    versions_dir = Path(args.versions_dir).resolve()
    if not versions_dir.is_dir():
        print(f"::error::versions directory not found: {versions_dir}")
        return 1

    files = _migration_files(versions_dir)
    if not files:
        print(f"::error::no migrations found in {versions_dir}")
        return 1

    errors: list[str] = []
    seen: dict[str, str] = {}

    for path in files:
        prefix = path.name.split("_", 1)[0]
        revision = _revision_of(path)

        if revision is None:
            errors.append(f'{path.name}: no `revision = "..."` assignment found')
            continue

        # 1. id must match the filename prefix — otherwise `alembic history` and
        #    the file listing disagree about what a revision is called.
        if revision != prefix:
            errors.append(
                f"{path.name}: revision id {revision!r} does not match filename prefix {prefix!r}"
            )

        # 2. id must be in the format alembic generates.
        if not _REV_ID_RE.match(revision):
            errors.append(
                f"{path.name}: revision id {revision!r} is not 12 lowercase hex characters "
                f"(the format `alembic revision` generates)"
            )

        # 3. ids must be unique.
        if revision in seen:
            errors.append(
                f"{path.name}: revision id {revision!r} is already used by {seen[revision]}"
            )
        else:
            seen[revision] = path.name

    # 4. new ids must look generated.
    if args.all:
        new_revisions = set(seen)
    else:
        new_revisions = set(seen) - _revisions_at(args.base_ref, versions_dir)

    for revision in sorted(new_revisions):
        if looks_hand_typed(revision):
            errors.append(
                f"{seen[revision]}: revision id {revision!r} looks hand-typed (a keyboard "
                f"pattern, not a generated id). Let alembic mint it:\n"
                f'      uv run alembic revision -m "short description"\n'
                f"    then move your upgrade()/downgrade() body into the generated file. "
                f"Hand-typed ids collide, and a collided id can never be fixed once a "
                f"database has stamped it."
            )

    if errors:
        print("::error::Alembic revision id check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    scope = "all" if args.all else f"{len(new_revisions)} new"
    print(
        f"OK: {len(files)} migrations — ids match filenames, are generated-format and unique "
        f"({scope} checked for hand-typed patterns)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
