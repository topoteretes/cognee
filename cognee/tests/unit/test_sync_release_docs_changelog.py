"""Changelog placement for the cognee-docs release sync (RES-37).

tools/ is not an importable package, so the module is loaded by path.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "sync_release_docs", TOOLS_DIR / "sync_release_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    # enhance_spec() reads tools/spec_extras.json relative to the module.
    sys.path.insert(0, str(TOOLS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS_DIR))
    return module


sync_release_docs = _load_module()


CHANGELOG_WITH_UNRELEASED = """---
title: "Changelog"
---

Cognee releases with highlights and links to the full release notes on GitHub.

## Unreleased

Changes queued for the next release.

---

## v1.5.4

Older release.
"""

CHANGELOG_WITHOUT_UNRELEASED = """---
title: "Changelog"
---

Cognee releases with highlights and links to the full release notes on GitHub.

## v1.5.4

Older release.
"""


def _headings(content: str) -> list[str]:
    return [line.strip() for line in content.splitlines() if line.startswith("## ")]


def _entry() -> str:
    return sync_release_docs.build_changelog_entry(
        "v1.5.5", "https://example.test/rel", "September 10, 2026", "Release notes."
    )


def test_entry_lands_below_unreleased_not_above_it():
    """The bug this pins: a release inserted above "Unreleased" reads as newer
    than unshipped work, leaving v1.5.5 -> Unreleased -> v1.5.4."""
    result = sync_release_docs.insert_entry_into_changelog(CHANGELOG_WITH_UNRELEASED, _entry())

    assert _headings(result) == ["## Unreleased", "## v1.5.5", "## v1.5.4"]


def test_unreleased_section_body_survives():
    result = sync_release_docs.insert_entry_into_changelog(CHANGELOG_WITH_UNRELEASED, _entry())

    assert "Changes queued for the next release." in result
    assert "Older release." in result
    assert result.startswith('---\ntitle: "Changelog"\n---\n')


def test_without_an_unreleased_section_the_entry_goes_first():
    result = sync_release_docs.insert_entry_into_changelog(CHANGELOG_WITHOUT_UNRELEASED, _entry())

    assert _headings(result) == ["## v1.5.5", "## v1.5.4"]


def test_intro_paragraph_is_never_swallowed():
    for existing in (CHANGELOG_WITH_UNRELEASED, CHANGELOG_WITHOUT_UNRELEASED):
        result = sync_release_docs.insert_entry_into_changelog(existing, _entry())
        assert "Cognee releases with highlights" in result


def test_a_changelog_with_no_entries_at_all_still_works():
    result = sync_release_docs.insert_entry_into_changelog(
        sync_release_docs.DEFAULT_CHANGELOG_TEXT, _entry()
    )

    assert _headings(result) == ["## v1.5.5"]
    assert "Cognee releases with highlights" in result


def test_repeated_sync_of_the_same_tag_is_a_no_op(tmp_path):
    changelog = tmp_path / "changelog.mdx"
    changelog.write_text(CHANGELOG_WITH_UNRELEASED, encoding="utf-8")

    kwargs = dict(
        tag="v1.5.5",
        release_url="https://example.test/rel",
        published_at="2026-09-10T12:00:00Z",
        release_body="Release notes.",
    )

    assert sync_release_docs.update_changelog_if_needed(changelog, **kwargs) is True
    after_first = changelog.read_text(encoding="utf-8")

    assert sync_release_docs.update_changelog_if_needed(changelog, **kwargs) is False
    assert changelog.read_text(encoding="utf-8") == after_first


@pytest.mark.parametrize("skip", [True, False])
def test_skip_changelog_flag_controls_whether_the_file_is_touched(tmp_path, skip, monkeypatch):
    """--skip-changelog exists so a preview run cannot fabricate a release
    section from its synthetic test-sync-<ref>-<sha> tag."""
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    changelog = docs_repo / "changelog.mdx"
    changelog.write_text(CHANGELOG_WITH_UNRELEASED, encoding="utf-8")

    spec_source = tmp_path / "spec.json"
    spec_source.write_text('{"openapi": "3.1.0"}', encoding="utf-8")
    body_file = tmp_path / "body.md"
    body_file.write_text("Preview run.", encoding="utf-8")

    argv = [
        "sync_release_docs.py",
        "--docs-repo",
        str(docs_repo),
        "--tag",
        "test-sync-4981-merge-639535e",
        "--release-url",
        "https://example.test/commit/639535e",
        "--published-at",
        "2026-09-08T15:00:00Z",
        "--release-body-file",
        str(body_file),
        "--openapi-output",
        str(spec_source),
        "--skip-openapi-generation",
    ]
    if skip:
        argv.append("--skip-changelog")
    monkeypatch.setattr(sys, "argv", argv)

    assert sync_release_docs.main() == 0

    content = changelog.read_text(encoding="utf-8")
    if skip:
        assert content == CHANGELOG_WITH_UNRELEASED
        assert "test-sync" not in content
    else:
        assert "## test-sync-4981-merge-639535e" in content
