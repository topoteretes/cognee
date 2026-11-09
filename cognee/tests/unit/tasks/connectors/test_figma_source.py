"""Unit tests for Cognee Figma connector."""

from unittest.mock import MagicMock

import pytest

try:
    from cognee.tasks.ingestion.connectors.figma import (
        FigmaAuthError,
        FigmaClient,
        FigmaError,
        FigmaRateLimitError,
        _extract_text_nodes,
        _iter_figma_files,
        build_figma_service,
        render_figma_page,
    )
except (ImportError, ModuleNotFoundError):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tasks/ingestion/connectors")))
    from figma import (
        FigmaAuthError,
        FigmaClient,
        FigmaError,
        FigmaRateLimitError,
        _extract_text_nodes,
        _iter_figma_files,
        build_figma_service,
        render_figma_page,
    )


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._json_data


def test_figma_client_auth_token():
    client = build_figma_service(access_token="figd_test_token_123")
    assert client.headers["X-Figma-Token"] == "figd_test_token_123"
    assert "Authorization" not in client.headers


def test_figma_client_auth_oauth():
    client = build_figma_service(oauth_token="oauth_test_token_456")
    assert client.headers["Authorization"] == "Bearer oauth_test_token_456"
    assert "X-Figma-Token" not in client.headers


def test_figma_client_missing_auth():
    with pytest.raises(ValueError, match="Provide either access_token or oauth_token"):
        FigmaClient()


def test_extract_text_nodes():
    tree = {
        "id": "0:1",
        "name": "Page 1",
        "type": "CANVAS",
        "children": [
            {
                "id": "1:2",
                "name": "Header Frame",
                "type": "FRAME",
                "children": [
                    {
                        "id": "1:3",
                        "name": "Title Text",
                        "type": "TEXT",
                        "characters": "Welcome to Design System",
                    },
                    {
                        "id": "1:4",
                        "name": "CTA Button",
                        "type": "TEXT",
                        "characters": "Get Started Now",
                    },
                ],
            }
        ],
    }

    text_nodes = []
    _extract_text_nodes(tree, text_nodes)

    assert len(text_nodes) == 2
    assert text_nodes[0]["text"] == "Welcome to Design System"
    assert text_nodes[1]["text"] == "Get Started Now"


def test_render_figma_page():
    components = [
        {"name": "PrimaryButton", "description": "Main call-to-action button with loading state."}
    ]
    text_nodes = [
        {"name": "Headline", "text": "Next-Gen AI Memory"}
    ]
    comments = [
        {"user": {"handle": "lead_designer"}, "message": "Updated contrast ratio to 4.5:1 for WCAG compliance."}
    ]

    rendered = render_figma_page(
        file_name="App Design v2",
        page_name="Onboarding Flow",
        page_id="0:1",
        components=components,
        text_nodes=text_nodes,
        comments=comments,
    )

    assert "Figma Design Document: App Design v2" in rendered
    assert "Page: Onboarding Flow (ID: 0:1)" in rendered
    assert "PrimaryButton: Main call-to-action button" in rendered
    assert '• [Headline]: "Next-Gen AI Memory"' in rendered
    assert "@lead_designer: Updated contrast ratio" in rendered


def test_iter_figma_files_emission():
    mock_session = MagicMock()

    file_payload = {
        "name": "Mobile Design Kit",
        "document": {
            "id": "0:0",
            "type": "DOCUMENT",
            "children": [
                {
                    "id": "0:1",
                    "name": "Home Screen",
                    "type": "CANVAS",
                    "children": [
                        {
                            "id": "2:1",
                            "name": "Hero Label",
                            "type": "TEXT",
                            "characters": "Explore Features",
                        }
                    ],
                }
            ],
        },
        "components": {
            "comp_1": {"name": "NavBar", "description": "Top fixed navigation bar"}
        },
    }

    comments_payload = {
        "comments": [
            {"user": {"handle": "sarah"}, "message": "Verified responsive layout"}
        ]
    }

    def fake_get(url, **kwargs):
        if url.endswith("/comments"):
            return FakeResponse(comments_payload, 200)
        return FakeResponse(file_payload, 200)

    mock_session.get.side_effect = fake_get

    client = FigmaClient(access_token="test_token", session=mock_session)
    rows = list(_iter_figma_files(client=client, file_keys=["testFileKey123"], include_comments=True))

    assert len(rows) == 1
    page_row = rows[0]
    assert page_row["id"] == "figma:testFileKey123:0:1"
    assert page_row["title"] == "Mobile Design Kit - Home Screen"
    assert "NavBar: Top fixed navigation bar" in page_row["content"]
    assert "Explore Features" in page_row["content"]
    assert "@sarah: Verified responsive layout" in page_row["content"]
    assert page_row["cognee_node_set"] == "figma:testFileKey123"
    assert page_row["document_source"] == "figma"


def test_figma_error_handling():
    mock_session = MagicMock()
    mock_session.get.return_value = FakeResponse({}, status_code=401)
    client = FigmaClient(access_token="invalid", session=mock_session)

    with pytest.raises(FigmaAuthError, match="Figma auth failed: HTTP 401"):
        client.fetch_file("fileKey")

    mock_session.get.return_value = FakeResponse({}, status_code=429)
    with pytest.raises(FigmaRateLimitError, match="Figma rate limit hit"):
        client.fetch_file("fileKey")


if __name__ == "__main__":
    print("Running Figma connector unit tests...")
    test_figma_client_auth_token()
    test_figma_client_auth_oauth()
    test_figma_client_missing_auth()
    test_extract_text_nodes()
    test_render_figma_page()
    test_iter_figma_files_emission()
    test_figma_error_handling()
    print("✅ All 7 Figma unit tests passed successfully!")
