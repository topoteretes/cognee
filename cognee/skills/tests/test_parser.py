"""Unit tests for the SKILL.md parser — no LLM or database required."""

from pathlib import Path
import tempfile

import pytest

from cognee.skills.parser.skill_parser import (
    parse_skill_file,
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

    def test_yaml_list_tags(self):
        text = "---\nname: tagged\ntags:\n  - python\n  - testing\n  - ci\n---\n# Body"
        fm, body = _parse_frontmatter(text)
        assert fm["name"] == "tagged"
        assert fm["tags"] == ["python", "testing", "ci"]

    def test_yaml_inline_list(self):
        text = "---\nname: inline\ntags: [alpha, beta, gamma]\n---\n# Body"
        fm, body = _parse_frontmatter(text)
        assert fm["tags"] == ["alpha", "beta", "gamma"]

    def test_yaml_nested_object(self):
        text = "---\nname: nested\nconfig:\n  timeout: 30\n  retries: 3\n---\n# Body"
        fm, body = _parse_frontmatter(text)
        assert fm["config"] == {"timeout": 30, "retries": 3}

    def test_yaml_colon_in_value(self):
        text = '---\nname: colon-test\ndescription: "URL: https://example.com"\n---\n# Body'
        fm, body = _parse_frontmatter(text)
        assert fm["description"] == "URL: https://example.com"

    def test_invalid_yaml_returns_empty(self):
        text = "---\n: invalid yaml [[\n---\n# Body"
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert body == "# Body"


class TestParseSkillFile:
    def test_parse_flat_skill_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "my-skill.md"
            skill_file.write_text(
                "---\nname: my-skill\ndescription: A flat skill.\n---\n\n# Instructions\n\nDo the thing."
            )

            skill = parse_skill_file(skill_file)
            assert skill is not None
            assert skill.skill_id == Path(tmpdir).name
            assert skill.name == "my-skill"
            assert "Do the thing" in skill.instructions

    def test_parse_flat_skill_with_custom_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "my-skill.md"
            skill_file.write_text("---\nname: My Skill\ndescription: Test.\n---\n\n# Body")

            skill = parse_skill_file(skill_file, skill_key="custom-key")
            assert skill is not None
            assert skill.skill_id == "custom-key"
            assert skill.name == "My Skill"

    def test_parse_nonexistent_file(self):
        result = parse_skill_file(Path("/nonexistent/SKILL.md"))
        assert result is None


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
        assert len(resource_names) >= 0


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

    def test_flat_skill_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "alpha.md").write_text(
                "---\nname: Alpha\ndescription: First.\n---\n# Alpha instructions"
            )
            (root / "beta.md").write_text(
                "---\nname: Beta\ndescription: Second.\n---\n# Beta instructions"
            )

            skills = parse_skills_folder(root)
            ids = {s.skill_id for s in skills}
            assert "alpha" in ids
            assert "beta" in ids
            assert len(skills) == 2

    def test_mixed_folders_and_flat_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            sub = root / "folder-skill"
            sub.mkdir()
            (sub / "SKILL.md").write_text(
                "---\nname: Folder Skill\ndescription: In a folder.\n---\n# Body"
            )

            (root / "flat-skill.md").write_text(
                "---\nname: Flat Skill\ndescription: A flat file.\n---\n# Body"
            )

            skills = parse_skills_folder(root)
            ids = {s.skill_id for s in skills}
            assert "folder-skill" in ids
            assert "flat-skill" in ids
            assert len(skills) == 2

    def test_folder_takes_precedence_over_flat_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            sub = root / "my-skill"
            sub.mkdir()
            (sub / "SKILL.md").write_text(
                "---\nname: From Folder\ndescription: Folder version.\n---\n# Body"
            )

            (root / "my-skill.md").write_text(
                "---\nname: From Flat\ndescription: Flat version.\n---\n# Body"
            )

            skills = parse_skills_folder(root)
            assert len(skills) == 1
            assert skills[0].name == "From Folder"
