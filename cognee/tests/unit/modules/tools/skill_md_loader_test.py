"""Tests for SKILL.md discovery: detection and directory loading must agree on
which files count as a SKILL.md regardless of filename case."""

import pytest

from cognee.modules.tools.ingest_skills import looks_like_skill_source
from cognee.modules.tools.loaders import iter_skill_md_files, load_skills_from_directory


SKILL_MD = """---
name: churn-analyst
description: Find patterns in customer churn.
allowed-tools: memory_search, load_skill
---
# Procedure
1. Search memory for churn signals.
"""


@pytest.mark.parametrize("file_name", ["SKILL.md", "skill.md", "Skill.md"])
def test_directory_detection_and_loading_agree(tmp_path, file_name):
    """A directory detected as a skill source must load at least one skill."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / file_name).write_text(SKILL_MD, encoding="utf-8")

    assert looks_like_skill_source(str(tmp_path)) is True

    skills = load_skills_from_directory(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "churn-analyst"
    assert skills[0].declared_tools == ["memory_search", "load_skill"]


def test_iter_skill_md_files_ignores_other_markdown(tmp_path):
    (tmp_path / "README.md").write_text("not a skill", encoding="utf-8")
    (tmp_path / "skills.md").write_text("not a skill either", encoding="utf-8")

    assert list(iter_skill_md_files(tmp_path)) == []
    assert looks_like_skill_source(str(tmp_path)) is False


def test_skill_md_file_detection_is_case_insensitive(tmp_path):
    skill_file = tmp_path / "skill.md"
    skill_file.write_text(SKILL_MD, encoding="utf-8")

    assert looks_like_skill_source(str(skill_file)) is True
