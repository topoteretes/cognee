"""Tests for prompt template rendering (SDK-203).

The Jinja2 environment previously enabled autoescape for ``.txt`` templates,
HTML-escaping every interpolated variable in every LLM prompt on the wire
(``'`` -> ``&#39;``, ``-->`` -> ``--&gt;``). Prompts must render content
verbatim; only markup templates (``.html``/``.xml``) stay escaped.
"""

from cognee.infrastructure.llm.prompts.render_prompt import render_prompt

RAW_TEXT = 'Zara\'s graph: a --> b & <c> "quoted"'


def test_txt_prompt_renders_content_verbatim():
    rendered = render_prompt(
        "context_for_question.txt",
        {"question": "What is Zara's plan?", "context": RAW_TEXT},
    )

    assert "What is Zara's plan?" in rendered
    assert RAW_TEXT in rendered
    for entity in ("&#39;", "&amp;", "&lt;", "&gt;", "&#34;", "&quot;"):
        assert entity not in rendered


def test_custom_base_directory_txt_template_not_escaped(tmp_path):
    (tmp_path / "custom.txt").write_text("value: {{ value }}", encoding="utf-8")

    rendered = render_prompt("custom.txt", {"value": RAW_TEXT}, base_directory=str(tmp_path))

    assert rendered == f"value: {RAW_TEXT}"


def test_html_templates_remain_autoescaped(tmp_path):
    # The markup-only autoescape posture is deliberate: .html/.xml templates
    # must keep escaping interpolated content.
    (tmp_path / "page.html").write_text("<p>{{ value }}</p>", encoding="utf-8")

    rendered = render_prompt(
        "page.html", {"value": "<script>alert(1)</script>"}, base_directory=str(tmp_path)
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
