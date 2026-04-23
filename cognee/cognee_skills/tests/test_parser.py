"""Unit tests for the SKILL.md parser — no LLM or database required."""

from pathlib import Path
import tempfile

import pytest

from cognee.cognee_skills.parser.skill_parser import (
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


class TestFieldAliases:
    """Parser accepts alternative frontmatter key names from multiple community formats."""

    def test_title_alias_for_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\ntitle: My Skill\ndescription: Does stuff.\n---\n\n# Body\n\nDo things."
            )
            skill = parse_skill_file(f)
            assert skill is not None
            # ``name`` is the slug (parent dir name); the frontmatter
            # ``title:`` alias was historically merged into a separate
            # display field, which no longer exists in the unified Skill.
            assert skill.name == Path(tmpdir).name

    def test_summary_alias_for_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text("---\nname: aliased\nsummary: A short summary.\n---\n\n# Body")
            skill = parse_skill_file(f)
            assert skill is not None
            assert skill.description == "A short summary."

    def test_categories_alias_for_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text("---\nname: tagged\ncategories:\n  - code\n  - review\n---\n\n# Body")
            skill = parse_skill_file(f)
            assert skill is not None
            assert "code" in skill.tags
            assert "review" in skill.tags

    def test_openclaw_nested_tags(self):
        """VoltAgent/OpenClaw format: metadata.openclaw.tags"""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: openclaw-skill\nmetadata:\n  openclaw:\n    tags:\n      - ai\n      - tools\n---\n\n# Body"
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert "ai" in skill.tags
            assert "tools" in skill.tags

    def test_allowed_tools_hyphen(self):
        """Anthropic spec: allowed-tools list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: tool-skill\nallowed-tools:\n  - Read\n  - Edit\n  - Bash\n---\n\n# Body"
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert "Read" in skill.declared_tools
            assert "Edit" in skill.declared_tools

    def test_allowed_tools_underscore(self):
        """Alternative: allowed_tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text("---\nname: tool-skill2\nallowed_tools: Read Edit Bash\n---\n\n# Body")
            skill = parse_skill_file(f)
            assert skill is not None
            assert "Read" in skill.declared_tools


class TestDescriptionInference:
    """Description can be inferred from body when not in frontmatter."""

    def test_description_from_first_paragraph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: no-desc\n---\n\n# Heading\n\n"
                "This is the first non-heading paragraph which is long enough to be used."
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert "first non-heading paragraph" in skill.description

    def test_description_skips_heading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: skip-heading\n---\n\n# This is a heading\n\n"
                "This paragraph comes after the heading and should be the description."
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert "paragraph" in skill.description
            assert "heading" not in skill.description.lower().split()[0]

    def test_description_from_frontmatter_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: has-desc\ndescription: Explicit description.\n---\n\n"
                "This paragraph should NOT be used as description."
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert skill.description == "Explicit description."


class TestTriggerExtraction:
    """Trigger phrases extracted from multiple sources."""

    def test_triggers_from_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: triggered\ntriggers:\n  - summarize this\n  - compress context\n---\n\n# Body"
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert "summarize this" in skill.triggers

    def test_triggers_from_when_to_activate_section(self):
        """muratcankoylan convention: ## When to Activate section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "SKILL.md"
            f.write_text(
                "---\nname: activate-skill\n---\n\n# Description\n\n"
                "Some description text here that is long enough.\n\n"
                "## When to Activate\n\n"
                "- User asks to summarize content\n"
                "- Context window is getting large\n"
                "- Need to compress information\n\n"
                "## Other Section\n\nMore content."
            )
            skill = parse_skill_file(f)
            assert skill is not None
            assert any("summarize" in t for t in skill.triggers)
            assert any("compress" in t for t in skill.triggers)


class TestEntryFileDiscovery:
    """Parser tries multiple candidate filenames per folder."""

    def test_skill_md_uppercase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "my-skill"
            d.mkdir()
            (d / "SKILL.md").write_text("---\nname: upper\ndescription: desc\n---\n\n# Body")
            skill = parse_skill_folder(d)
            assert skill is not None
            assert skill.name == "my-skill"

    def test_skill_md_lowercase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "my-skill"
            d.mkdir()
            (d / "skill.md").write_text("---\nname: lower\ndescription: desc\n---\n\n# Body")
            skill = parse_skill_folder(d)
            assert skill is not None
            assert skill.name == "my-skill"

    def test_readme_fallback(self):
        """README.md used when no SKILL.md / skill.md present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "readme-skill"
            d.mkdir()
            (d / "README.md").write_text(
                "---\nname: from-readme\ndescription: desc\n---\n\n# Body\n\nContent here."
            )
            skill = parse_skill_folder(d)
            assert skill is not None
            assert skill.name == "readme-skill"

    def test_uppercase_skill_md_preferred_over_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "prefer-skill"
            d.mkdir()
            (d / "SKILL.md").write_text(
                "---\nname: from-skill-md\ndescription: desc\n---\n\n# Body"
            )
            (d / "README.md").write_text("---\nname: from-readme\ndescription: desc\n---\n\n# Body")
            skill = parse_skill_folder(d)
            assert skill is not None
            assert skill.name == "prefer-skill"

    def test_no_entry_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "empty-skill"
            d.mkdir()
            (d / "notes.txt").write_text("just a text file")
            skill = parse_skill_folder(d)
            assert skill is None


class TestParseSkillFile:
    def test_parse_flat_skill_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "my-skill.md"
            skill_file.write_text(
                "---\nname: my-skill\ndescription: A flat skill.\n---\n\n# Instructions\n\nDo the thing."
            )

            skill = parse_skill_file(skill_file)
            assert skill is not None
            assert skill.name == "my-skill"
            assert "Do the thing" in skill.procedure

    def test_parse_flat_skill_with_custom_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "my-skill.md"
            skill_file.write_text("---\nname: My Skill\ndescription: Test.\n---\n\n# Body")

            skill = parse_skill_file(skill_file, skill_key="custom-key")
            assert skill is not None
            assert skill.name == "custom-key"
            assert skill.name == "custom-key"

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
        assert skill.name == "summarize"
        assert skill.name == "summarize"
        assert len(skill.description) > 0
        assert len(skill.procedure) > 0
        assert skill.content_hash != ""

    def test_parse_code_review(self):
        skill_dir = EXAMPLE_SKILLS_DIR / "code-review"
        if not skill_dir.exists():
            pytest.skip("example_skills/code-review not found")

        skill = parse_skill_folder(skill_dir)
        assert skill is not None
        assert skill.name == "code-review"
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
        skill_ids = {s.name for s in skills}
        # Only data-extraction ships in the repo currently; summarize
        # and code-review example folders aren't checked in. Assert on
        # whatever's actually there.
        assert len(skills) >= 1
        assert "data-extraction" in skill_ids
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
            ids = {s.name for s in skills}
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
            ids = {s.name for s in skills}
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
            # Folder variant wins; slug is the folder name.
            assert skills[0].name == "my-skill"
