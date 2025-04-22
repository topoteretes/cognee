from typing import Union, Optional, Type, List
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet
from cognee.shared.data_models import Edge
from pydantic import BaseModel, Field
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.config import get_llm_config
from uuid import UUID


class AssociativeEdge(BaseModel):
    source_node: str
    target_node: str
    relationship_name: str
    reason: str


class AssociativeEdges(BaseModel):
    edges: List[AssociativeEdge] = Field(..., default_factory=list)


async def node_set_edge_association():
    graph_engine = await get_graph_engine()

    node_set_names = await graph_engine.query("""MATCH (n)
                                   WHERE n.type = 'NodeSet'
                                   RETURN n.name AS name
                                   """)

    for node_set in node_set_names:
        node_name = node_set.get("name", None)
        nodes_data, edges_data = await graph_engine.get_subgraph(
            node_type=NodeSet, node_name=node_name
        )
        nodes = {}
        for node_id, attributes in nodes_data:
            if node_id not in nodes:
                text = attributes.get("text")
                if text:
                    name = text.strip().split("\n")[0][:50]
                    content = text
                else:
                    name = attributes.get("name", "Unnamed Node")
                    content = name
                nodes[node_id] = {"node": attributes, "name": name, "content": content}

        name_to_uuid = {data["name"].strip().lower(): node_id for node_id, data in nodes.items()}

        subgraph_description = create_subgraph_description(nodes, edges_data)

        llm_client = get_llm_client()

        system_prompt = render_prompt("edge_association_prompt.txt", {})
        associative_edges = await llm_client.acreate_structured_output(
            subgraph_description, system_prompt, AssociativeEdges
        )

        graph_edges = []
        for ae in associative_edges.edges:
            src_str = name_to_uuid.get(ae.source_node)
            tgt_str = name_to_uuid.get(ae.target_node)
            if not src_str or not tgt_str:
                continue

            src = UUID(src_str)
            tgt = UUID(tgt_str)
            rel = ae.relationship_name
            rea = ae.reason

            props = {
                "ontology_valid": False,
                "relationship_name": rel,
                "source_node_id": src,
                "target_node_id": tgt,
                "reason": rea,
            }

            graph_edges.append((src, tgt, rel, props))

        if graph_edges:
            await graph_engine.add_edges(graph_edges)

        print()


def create_subgraph_description(nodes, edges_data):
    node_section = "\n".join(
        f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
        for info in nodes.values()
    )

    connection_section = "\n".join(
        f"{nodes[source_id]['name']} --[{relationship_type}]--> {nodes[target_id]['name']}"
        for source_id, target_id, relationship_type, attributes in edges_data
        if source_id in nodes and target_id in nodes
    )

    return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"
