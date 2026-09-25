"""Tests for reading prompt files (SDK-804).

Prompt names can come from callers (``system_prompt_path``), so a name that
resolves outside its prompt directory must be refused rather than read.
"""

import os

import pytest

from cognee.infrastructure.llm.exceptions import PromptPathOutsideBaseDirectoryError
from cognee.infrastructure.llm.prompts.read_query_prompt import read_query_prompt


def test_default_prompt_name_reads():
    assert read_query_prompt("answer_simple_question.txt")


def test_missing_prompt_returns_none():
    assert read_query_prompt("does_not_exist.txt") is None


def test_custom_base_directory_reads(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "custom.txt").write_text("custom prompt", encoding="utf-8")

    assert read_query_prompt("nested/custom.txt", base_directory=str(tmp_path)) == "custom prompt"


def test_absolute_path_is_refused(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("not a prompt", encoding="utf-8")

    with pytest.raises(PromptPathOutsideBaseDirectoryError):
        read_query_prompt(str(outside))


def test_parent_traversal_is_refused(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    (tmp_path / "outside.txt").write_text("not a prompt", encoding="utf-8")

    with pytest.raises(PromptPathOutsideBaseDirectoryError):
        read_query_prompt("../outside.txt", base_directory=str(base))


def test_symlink_out_of_base_is_refused(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not a prompt", encoding="utf-8")
    os.symlink(outside, base / "link.txt")

    with pytest.raises(PromptPathOutsideBaseDirectoryError):
        read_query_prompt("link.txt", base_directory=str(base))


def test_base_directory_itself_is_refused(tmp_path):
    with pytest.raises(PromptPathOutsideBaseDirectoryError):
        read_query_prompt(".", base_directory=str(tmp_path))
