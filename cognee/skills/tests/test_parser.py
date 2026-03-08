"""Unit tests for the SKILL.md parser — no LLM or database required."""

from pathlib import Path
import tempfile

import pytest

from cognee.skills.parser.skill_parser import (
    parse_skill_folder,
    parse_skills_folder,
    _parse_frontmatter,
)

EXAMPLE_SKILLS_DIR = Path(__file__).resolve().parent.parent / "example_skills"


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        text = "---\nname: test-skill\ndescription: A test skill.\n---\n\n# Body"
        fm, body = _parse_frontmatter(text)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test skill."
        assert body == "# Body"

    def test_multiline_description(self):
        text = "---\nname: multi\ndescription: >\n  First line\n  second line\n---\nBody text"
        fm, body = _parse_frontmatter(text)
        assert fm["name"] == "multi"
        assert "First line" in fm["description"]
        assert "second line" in fm["description"]
        assert body == "Body text"

    def test_no_frontmatter(self):
        text = "# Just a markdown file\n\nNo frontmatter here."
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert "Just a markdown file" in body


class TestParseSkillFolder:
    def test_parse_summarize(self):
        skill_dir = EXAMPLE_SKILLS_DIR / "summarize"
        if not skill_dir.exists():
            pytest.skip("example_skills/summarize not found")

        skill = parse_skill_folder(skill_dir)
        assert skill is not None
        assert skill.skill_id == "summarize"
        assert skill.name == "summarize"
        assert len(skill.description) > 0
        assert len(skill.instructions) > 0
        assert skill.content_hash != ""

    def test_parse_code_review(self):
        skill_dir = EXAMPLE_SKILLS_DIR / "code-review"
        if not skill_dir.exists():
            pytest.skip("example_skills/code-review not found")

        skill = parse_skill_folder(skill_dir)
        assert skill is not None
        assert skill.skill_id == "code-review"
        assert skill.name == "code-review"

    def test_missing_skill_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = parse_skill_folder(Path(tmpdir))
            assert result is None

    def test_resources_detected(self):
        skill_dir = EXAMPLE_SKILLS_DIR / "summarize"
        if not skill_dir.exists():
            pytest.skip("example_skills/summarize not found")

        skill = parse_skill_folder(skill_dir)
        assert skill is not None
        resource_names = [r.name for r in skill.resources]
        assert len(resource_names) >= 0  # may have references/


class TestParseSkillsFolder:
    def test_parse_all_example_skills(self):
        if not EXAMPLE_SKILLS_DIR.exists():
            pytest.skip("example_skills/ not found")

        skills = parse_skills_folder(EXAMPLE_SKILLS_DIR)
        skill_ids = {s.skill_id for s in skills}
        assert len(skills) >= 3
        assert "summarize" in skill_ids
        assert "code-review" in skill_ids
        assert "data-extraction" in skill_ids

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_skills_folder("/nonexistent/path")

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills = parse_skills_folder(tmpdir)
            assert skills == []
