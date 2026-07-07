from cognee.modules.graph.graph_report import (
    build_graph_report,
    render_graph_report_markdown,
    suggest_questions_from_hubs,
)


def _sample_graph():
    nodes = [
        (
            "alpha",
            {
                "type": "Entity",
                "name": "Alpha",
                "source_node_set": "marketing, strategy",
                "source_content_hash": "doc-alpha",
            },
        ),
        (
            "beta",
            {
                "type": "Entity",
                "name": "Beta",
                "source_node_set": "sales",
                "source_content_hash": "doc-beta",
            },
        ),
        (
            "gamma",
            {
                "type": "Entity",
                "name": "Gamma",
                "source_node_set": "marketing",
                "source_content_hash": "doc-alpha",
            },
        ),
        (
            "delta",
            {
                "type": "DocumentChunk",
                "name": "Delta chunk",
                "belongs_to_set": ["support"],
                "source_content_hash": "doc-delta",
            },
        ),
    ]
    edges = [
        ("alpha", "beta", "partners_with", {"ontology_valid": True}),
        ("alpha", "gamma", "mentions", {}),
        (
            "alpha",
            "delta",
            "associated_with",
            {"ontology_valid": False, "reasoning": "similar chunks"},
        ),
        ("alpha", "alpha", "SELF", {}),
    ]
    return nodes, edges


def test_graph_report_ranks_hubs_and_ignores_self_loops():
    report = build_graph_report(_sample_graph(), top_n=3)

    assert report["summary"]["node_count"] == 4
    assert report["summary"]["edge_count"] == 3
    assert report["hub_nodes"][0]["id"] == "alpha"
    assert report["hub_nodes"][0]["degree"] == 3
    assert report["hub_nodes"][0]["node_sets"] == ["marketing", "strategy"]


def test_graph_report_surprising_connections_and_confidence_tags():
    report = build_graph_report(_sample_graph(), top_n=5)

    confidence_tags = report["confidence_tags"]
    assert confidence_tags["EXTRACTED"] == 2
    assert confidence_tags["INFERRED"] == 1

    surprising = report["surprising_connections"]
    assert {item["relation"] for item in surprising} == {"associated_with", "partners_with"}
    assert surprising[0]["novelty_score"] >= surprising[-1]["novelty_score"]
    assert any("different node sets" in reason for reason in surprising[0]["reasons"])


def test_graph_report_can_scope_by_node_set():
    report = build_graph_report(_sample_graph(), top_n=5, node_name=["sales"])

    assert report["summary"]["scoped_node_sets"] == ["sales"]
    assert report["summary"]["edge_count"] == 1
    assert {hub["id"] for hub in report["hub_nodes"]} == {"alpha", "beta"}


def test_graph_report_markdown_and_questions_are_readable():
    report = build_graph_report(
        _sample_graph(),
        top_n=2,
        suggested_questions=["What connects Alpha and Beta?"],
    )
    markdown = render_graph_report_markdown(report)

    assert "# Graph Insight Report" in markdown
    assert "## Hub Nodes" in markdown
    assert "Alpha" in markdown
    assert "What connects Alpha and Beta?" in markdown

    fallback_questions = suggest_questions_from_hubs(report["hub_nodes"])
    assert fallback_questions[0] == "What makes Alpha central in this knowledge graph?"
