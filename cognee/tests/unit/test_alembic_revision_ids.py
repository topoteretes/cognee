"""Alembic revision ids must be generated, unique, and match their filenames.

A revision id is the value stamped into every deployed database's
``alembic_version`` table, so it can never be changed after a release ships.
Hand-typed ids collide — ``b2c3d4e5f6a7`` already names two different migrations
across this repo and the Cloud migration trees — and a collision means one id
describes two schemas.

The invariants below run in the normal unit-test job. The companion check for
*newly added* ids that look hand-typed needs git, so it lives in
``tools/check_alembic_revision_ids.py`` and runs from pre-commit.
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VERSIONS_DIR = REPO_ROOT / "cognee" / "alembic" / "versions"

sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_alembic_revision_ids import (  # noqa: E402
    _REV_ID_RE,
    _migration_files,
    _revision_of,
    looks_hand_typed,
)


class TestAlembicRevisionIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = _migration_files(VERSIONS_DIR)

    def test_versions_directory_is_not_empty(self):
        """A silently empty glob would make every assertion below vacuous."""
        self.assertTrue(self.files, f"no migrations found in {VERSIONS_DIR}")

    def test_revision_id_matches_filename_prefix(self):
        """`alembic history` and the file listing must name a revision the same way."""
        for path in self.files:
            with self.subTest(migration=path.name):
                revision = _revision_of(path)
                self.assertIsNotNone(revision, f"{path.name}: no `revision = ...` found")
                self.assertEqual(
                    revision,
                    path.name.split("_", 1)[0],
                    f"{path.name}: revision id does not match the filename prefix",
                )

    def test_revision_ids_are_in_generated_format(self):
        """12 lowercase hex characters — what alembic's rev_id() produces."""
        for path in self.files:
            with self.subTest(migration=path.name):
                revision = _revision_of(path)
                self.assertRegex(
                    revision or "",
                    _REV_ID_RE,
                    f"{path.name}: {revision!r} is not 12 lowercase hex characters. "
                    "Let `alembic revision -m ...` mint the id.",
                )

    def test_revision_ids_are_unique(self):
        """Two migrations sharing an id makes `alembic_version` ambiguous."""
        seen: dict[str, str] = {}
        for path in self.files:
            revision = _revision_of(path)
            if revision is None:
                continue
            self.assertNotIn(
                revision,
                seen,
                f"{path.name}: revision id {revision!r} is already used by {seen.get(revision)}",
            )
            seen[revision] = path.name

    def test_down_revisions_resolve_and_form_one_chain(self):
        """Every down_revision must exist, and the graph must have a single head.

        A dangling parent makes `alembic upgrade head` fail at runtime, in the
        deployment rather than in CI.
        """
        down_re = re.compile(r"^down_revision(?:\s*:\s*[^=]+)?\s*=\s*(.+)$", re.MULTILINE)
        parents: dict[str, list[str]] = {}
        for path in self.files:
            revision = _revision_of(path)
            if revision is None:
                continue
            match = down_re.search(path.read_text(encoding="utf-8"))
            raw = match.group(1).split("#")[0] if match else ""
            parents[revision] = re.findall(r"['\"]([^'\"]+)['\"]", raw)

        for revision, downs in parents.items():
            for down in downs:
                self.assertIn(
                    down,
                    parents,
                    f"{revision}: down_revision {down!r} does not exist in the versions directory",
                )

        referenced = {down for downs in parents.values() for down in downs}
        heads = sorted(set(parents) - referenced)
        self.assertEqual(len(heads), 1, f"expected exactly one head, found {len(heads)}: {heads}")

    def test_hand_typed_detector_accepts_generated_ids(self):
        """The detector must never reject an id alembic actually generated."""
        from uuid import uuid4

        generated = [uuid4().hex[-12:] for _ in range(5000)]
        false_positives = [rev for rev in generated if looks_hand_typed(rev)]
        self.assertEqual(false_positives, [], "detector rejected generated ids")

    def test_hand_typed_detector_catches_keyboard_patterns(self):
        """Regression pins for the id family that caused the Cloud collision."""
        for revision in ("a1b2c3d4e5f6", "b2c3d4e5f6a7", "d4e5f6a7b8c9", "e5f6a7b8c9d0"):
            with self.subTest(revision=revision):
                self.assertTrue(looks_hand_typed(revision))


if __name__ == "__main__":
    unittest.main()
