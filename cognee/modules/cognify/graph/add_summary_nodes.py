"""Here we update the graph with content summary that summarizer produced"""

async def add_summary_nodes(graph_client, document_id, summary):
    summary_node_id = f"DATA_SUMMARY__{document_id}"

    await graph_client.add_node(
        summary_node_id,
        dict(
            name = "Summary",
            summary = summary["summary"],
        ),
    )

    await graph_client.add_edge(document_id, summary_node_id, relationship_name = "summarized_as")


    description_node_id = f"DATA_DESCRIPTION__{document_id}"

    await graph_client.add_node(
        description_node_id,
        dict(
            name = "Description",
            summary = summary["description"],
        ),
    )

    await graph_client.add_edge(document_id, description_node_id, relationship_name = "described_as")
