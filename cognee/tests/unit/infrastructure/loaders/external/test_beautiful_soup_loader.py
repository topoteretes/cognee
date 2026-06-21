from cognee.infrastructure.loaders.external.beautiful_soup_loader import (
    BeautifulSoupLoader,
)


def test_deduplicate_content_removes_duplicates():
    loader = BeautifulSoupLoader()

    pieces = [
        "Cognee Docs Store and retrieve knowledge.",
        "Cognee Docs Store and retrieve knowledge.",
        "Installation pip install cognee",
    ]

    result = loader._deduplicate_content(pieces)

    assert result == [
        "Cognee Docs Store and retrieve knowledge.",
        "Installation pip install cognee",
    ]


def test_deduplicate_content_preserves_unique_content():
    loader = BeautifulSoupLoader()

    pieces = [
        "Main content",
        "Documentation",
        "Installation",
    ]

    result = loader._deduplicate_content(pieces)

    assert result == pieces