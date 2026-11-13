"""
Unit tests for the arXiv data-source connector.
All HTTP calls are mocked — no internet or API key required.
"""

import pytest
from unittest.mock import MagicMock, patch

from cognee.tasks.ingestion.connectors.arxiv import (
    _parse_arxiv_xml,
    _paper_to_document,
    _fetch_papers,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex
recurrent or convolutional neural networks that include an encoder and a decoder.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.LG"/>
    <link rel="related" title="pdf" href="https://arxiv.org/pdf/1706.03762"/>
    <arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>
  </entry>
</feed>"""

EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
</feed>"""

MULTI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>Abstract one.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <link rel="related" title="pdf" href="https://arxiv.org/pdf/1706.03762"/>
    <arxiv:doi></arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.07041v1</id>
    <title>A Second Paper</title>
    <summary>Abstract two.</summary>
    <published>2023-01-17T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
    <arxiv:primary_category term="cs.AI"/>
    <link rel="related" title="pdf" href="https://arxiv.org/pdf/2301.07041"/>
  </entry>
</feed>"""


def _make_mock_client(xml_body: str):
    """Synchronous mock httpx.Client."""
    mock_response = MagicMock()
    mock_response.text = xml_body
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# _parse_arxiv_xml
# ---------------------------------------------------------------------------

def test_parse_single_paper():
    papers = _parse_arxiv_xml(SAMPLE_XML)
    assert len(papers) == 1

    p = papers[0]
    assert "Attention Is All You Need" in p["title"]
    assert "Ashish Vaswani" in p["authors"]
    assert "Noam Shazeer" in p["authors"]
    assert "cs.CL" in p["categories"]
    assert "cs.LG" in p["categories"]
    assert p["published"] == "2017-06-12"
    assert "1706.03762" in p["pdf_url"]
    assert p["doi"] == "10.48550/arXiv.1706.03762"


def test_parse_multiple_papers():
    papers = _parse_arxiv_xml(MULTI_XML)
    assert len(papers) == 2
    assert "Attention Is All You Need" in papers[0]["title"]
    assert "A Second Paper" in papers[1]["title"]


def test_parse_empty_feed():
    papers = _parse_arxiv_xml(EMPTY_XML)
    assert papers == []


def test_categories_deduplicated():
    """Primary category must not appear twice even when also listed in <category>."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>Dedup Test</title>
    <summary>Abstract.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Author One</name></author>
    <arxiv:primary_category term="cs.AI"/>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
  </entry>
</feed>"""
    papers = _parse_arxiv_xml(xml)
    cats = papers[0]["categories"]
    assert cats.count("cs.AI") == 1
    assert "cs.LG" in cats
    assert cats[0] == "cs.AI"  # primary comes first


def test_title_newlines_stripped():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/9999.0001</id>
    <title>A Long Title
That Wraps Onto
Multiple Lines</title>
    <summary>Abstract.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Author</name></author>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
</feed>"""
    papers = _parse_arxiv_xml(xml)
    assert "\n" not in papers[0]["title"]


# ---------------------------------------------------------------------------
# _paper_to_document
# ---------------------------------------------------------------------------

def test_paper_to_document_contains_required_fields():
    paper = {
        "id": "http://arxiv.org/abs/1706.03762",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models...",
        "published": "2017-06-12",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "categories": ["cs.CL", "cs.LG"],
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "doi": "10.48550/arXiv.1706.03762",
    }
    doc = _paper_to_document(paper)

    assert "Attention Is All You Need" in doc
    assert "Ashish Vaswani" in doc
    assert "Noam Shazeer" in doc
    assert "cs.CL" in doc
    assert "Abstract" in doc
    assert "Source:** arxiv" in doc
    assert "2017-06-12" in doc
    assert "10.48550/arXiv.1706.03762" in doc


def test_paper_to_document_no_doi():
    paper = {
        "id": "http://arxiv.org/abs/9999.9999",
        "title": "No DOI Paper",
        "abstract": "Abstract here.",
        "published": "2024-01-01",
        "authors": ["Jane Doe"],
        "categories": ["cs.AI"],
        "pdf_url": "",
        "doi": "",
    }
    doc = _paper_to_document(paper)
    assert "**DOI:**" not in doc


def test_paper_to_document_no_authors():
    paper = {
        "id": "http://arxiv.org/abs/9999.9998",
        "title": "No Authors",
        "abstract": "Abstract.",
        "published": "2024-01-01",
        "authors": [],
        "categories": ["cs.AI"],
        "pdf_url": "",
        "doi": "",
    }
    doc = _paper_to_document(paper)
    assert "Unknown" in doc


# ---------------------------------------------------------------------------
# _fetch_papers
# ---------------------------------------------------------------------------

def test_fetch_by_query():
    client = _make_mock_client(SAMPLE_XML)
    papers = _fetch_papers(client, query="attention transformer", paper_ids=None, category=None, max_results=1)

    client.get.assert_called_once()
    call_params = client.get.call_args[1]["params"]
    assert "all:attention transformer" in call_params["search_query"]
    assert papers[0]["title"] == "Attention Is All You Need"


def test_fetch_by_paper_ids():
    client = _make_mock_client(SAMPLE_XML)
    papers = _fetch_papers(client, query="", paper_ids=["1706.03762"], category=None, max_results=1)

    call_params = client.get.call_args[1]["params"]
    assert "1706.03762" in call_params["id_list"]
    assert len(papers) == 1


def test_fetch_by_category():
    client = _make_mock_client(SAMPLE_XML)
    _fetch_papers(client, query="", paper_ids=None, category="cs.AI", max_results=5)

    call_params = client.get.call_args[1]["params"]
    assert "cat:cs.AI" in call_params["search_query"]


def test_fetch_query_and_category_combined():
    client = _make_mock_client(SAMPLE_XML)
    _fetch_papers(client, query="diffusion", paper_ids=None, category="cs.LG", max_results=5)

    call_params = client.get.call_args[1]["params"]
    assert "all:diffusion" in call_params["search_query"]
    assert "cat:cs.LG" in call_params["search_query"]


def test_fetch_empty_result():
    client = _make_mock_client(EMPTY_XML)
    papers = _fetch_papers(client, query="nonexistentxyz123456789", paper_ids=None, category=None, max_results=1)
    assert papers == []


def test_fetch_multiple_papers():
    client = _make_mock_client(MULTI_XML)
    papers = _fetch_papers(client, query="test", paper_ids=None, category=None, max_results=2)
    assert len(papers) == 2
