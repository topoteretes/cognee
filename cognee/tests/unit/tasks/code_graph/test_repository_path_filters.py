"""User path selection shares the exact preview and checkout rules."""

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.tasks.code_graph.config import validate_codegraph_config
from cognee.tasks.code_graph.repo_selection import (
    path_matches,
    preview_tree,
    select_tree,
    validate_path_patterns,
)


def tree(*paths):
    return "".join(f"100644 blob {'a' * 40}\t{path}\0" for path in paths)


@pytest.mark.parametrize(
    "path, pattern, expected",
    [
        ("main.py", "**/*.py", True),
        ("src/nested/main.py", "src/**/*.py", True),
        ("src/main.py", "src/**/*.py", True),
        ("src/nested/main.py", "src/*.py", False),
        ("src/Main.py", "src/main.py", False),
        ("docs/a.md", "docs/[ab].?d", True),
        ("docs/a.md", "docs/**", True),
        ("src/main.py", "src/*.py", True),
        ("src/a/b.py", "src/*/b.py", True),
        ("src/a/b/c.py", "src/*/c.py", False),
        ("src", "src/*.py", False),
    ],
)
def test_posix_pattern_semantics(path, pattern, expected):
    assert path_matches(path, pattern) is expected


def test_preview_and_selection_agree_and_exclusions_win():
    listing = tree("main.py", "src/a.py", "src/tests/a.py", "README.md", "weights.bin")
    preview = preview_tree(listing, False, ["**/*.py"], ["**/tests/**"])
    assert preview["selected"] == ["main.py", "src/a.py"]
    assert preview["skipped"]["excluded"] == ["src/tests/a.py"]
    assert preview["skipped"]["not_included"] == ["README.md"]
    assert preview["skipped"]["binary"] == ["weights.bin"]
    selected, counts = select_tree(listing, False, ["**/*.py"], ["**/tests/**"])
    assert selected == preview["selected"]
    assert counts == {reason: len(paths) for reason, paths in preview["skipped"].items()}


def test_empty_selection_is_not_all():
    assert select_tree(tree("main.py"), False, [])[0] == []
    assert select_tree(tree("main.py"), False, None)[0] == ["main.py"]


@pytest.mark.parametrize(
    "pattern",
    ["", "/src/**", "../src", "src/../a", "a\\b", "a//b", "./a", "C:/src", "a/**x", "a\nb"],
)
def test_invalid_patterns_fail_before_selection(pattern):
    with pytest.raises(CogneeValidationError):
        validate_codegraph_config({"include_paths": [pattern]})
    with pytest.raises(CogneeValidationError):
        select_tree("", False, exclude_paths=[pattern])


@pytest.mark.parametrize("value", ["src/**", [1], {}, True])
def test_pattern_types_are_strict(value):
    with pytest.raises(CogneeValidationError):
        validate_codegraph_config({"exclude_paths": value})


@pytest.mark.asyncio
async def test_local_directory_filters_raise_without_modifying_files(tmp_path):
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError, resolve_repo_source

    source = tmp_path / "main.py"
    source.write_text("keep = True\n")
    with pytest.raises(CodeRepositoryError, match="require a remote repository URL"):
        await resolve_repo_source(tmp_path, include_paths=[])
    assert source.read_text() == "keep = True\n"


def test_none_is_a_filter_only_where_the_caller_says_so():
    validate_path_patterns(None, "include_paths", optional=True)
    for name in ("include_paths", "exclude_paths"):
        with pytest.raises(CogneeValidationError):
            validate_path_patterns(None, name)


def test_selection_counts_what_preview_lists():
    listing = tree("a.py", "docs/b.md", "node_modules/c.js", "logo.png")
    selected, counts = select_tree(listing, False, ["**/*.py"], ["docs/**"])
    preview = preview_tree(listing, False, ["**/*.py"], ["docs/**"])
    assert selected == preview["selected"]
    assert counts == {reason: len(paths) for reason, paths in preview["skipped"].items()}


def test_the_classifier_validates_when_called_not_when_first_consumed():
    from cognee.tasks.code_graph.repo_selection import _classify_tree

    with pytest.raises(CogneeValidationError):
        _classify_tree(tree("a.py"), False, ["/absolute"], None)
    with pytest.raises(Exception, match="incomplete"):
        _classify_tree("100644 blob " + "a" * 40 + "\ta.py", False, None, None)
