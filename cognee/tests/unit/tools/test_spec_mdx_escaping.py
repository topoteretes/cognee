"""Unit tests for the MDX escaping `enhance_spec` applies to spec descriptions.

Mintlify compiles OpenAPI `description` fields as MDX, where `{` opens a JS
expression and `<` opens a JSX tag. An unescaped one is a syntax error, and
Mintlify degrades the whole description to raw text instead of failing, so the
published page shows literal `##` and `**` markup. See `_escape_mdx`.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))

from sync_release_docs import _escape_mdx, escape_descriptions


@pytest.mark.parametrize(
    "text",
    [
        'e.g. [{"source": "crm"}, null], paired positionally',
        "- Single pipeline (default): {dataset_id: status}",
        "- Multiple pipelines: {dataset_id: {pipeline_name: status}}",
        '"total": <int>,',
        "values starting with '[' or '{' are rejected",
        'Returns a JSON list of users: [{"id", "email"}].',
    ],
)
def test_mdx_significant_characters_are_escaped(text):
    escaped = _escape_mdx(text)
    prose = re.sub(r"\\[{}<]", "", escaped)
    assert "{" not in prose and "}" not in prose
    assert not re.search(r"<(?=[A-Za-z/])", prose)


@pytest.mark.parametrize(
    "text",
    [
        "the `{dataset_id}` path parameter",
        "``GET /tenants/{id}/users`` uses this scope",
        'a fence:\n\n```\n{\n  "a": <int>\n}\n```\n',
    ],
)
def test_code_spans_are_left_alone(text):
    # MDX does not interpret code contents, and a backslash inside a span or fence
    # renders as a literal backslash.
    assert _escape_mdx(text) == text


def test_escaping_is_idempotent():
    # `enhance_spec` runs on a freshly generated spec, so a description should never
    # be escaped twice — but a second pass must not produce `\\{`, which would render
    # as a visible backslash.
    once = _escape_mdx('e.g. [{"source": "crm"}, null]')
    assert _escape_mdx(once) == once


def test_escape_descriptions_walks_the_whole_document():
    spec = {
        "paths": {
            "/api/v1/add": {"post": {"description": 'e.g. {"source": "crm"}'}},
        },
        "components": {
            "schemas": {
                "Body": {
                    "properties": {
                        "external_metadata": {"description": "objects like {} or {a: 1}"}
                    }
                }
            }
        },
        "servers": [{"url": "https://api.cognee.ai", "description": "Production server"}],
    }
    escape_descriptions(spec)
    assert spec["paths"]["/api/v1/add"]["post"]["description"] == ('e.g. \\{"source": "crm"\\}')
    assert (
        spec["components"]["schemas"]["Body"]["properties"]["external_metadata"]["description"]
        == "objects like \\{\\} or \\{a: 1\\}"
    )
    # A description with nothing MDX-significant in it is untouched.
    assert spec["servers"][0]["description"] == "Production server"
