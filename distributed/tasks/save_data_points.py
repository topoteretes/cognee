# import json
# import asyncio
from pympler import asizeof

# from cognee.modules.storage.utils import JSONEncoder
from distributed.queues import save_data_points_queue
# from cognee.modules.graph.utils import get_graph_from_model


async def save_data_points(data_points_and_relationships: tuple[list, list]):
    # data_points = data_points_and_relationships[0]
    # data_point_connections = data_points_and_relationships[1]

    # added_nodes = {}
    # added_edges = {}
    # visited_properties = {}

    # nodes_and_edges: list[tuple] = await asyncio.gather(
    #     *[
    #         get_graph_from_model(
    #             data_point,
    #             added_nodes=added_nodes,
    #             added_edges=added_edges,
    #             visited_properties=visited_properties,
    #         )
    #         for data_point in data_points
    #     ]
    # )

    # graph_data_deduplication = GraphDataDeduplication()
    # deduplicated_nodes_and_edges = [graph_data_deduplication.deduplicate_nodes_and_edges(nodes, edges + data_point_connections) for nodes, edges in nodes_and_edges]

    node_batch = []
    edge_batch = []

    for nodes, edges in data_points_and_relationships:
        for node in nodes:
            if asizeof.asizeof(node) >= 500000:
                try_pushing_nodes_to_queue([node])
                continue
                # print(f"Node too large:\n{node.id}\n")

            node_batch.append(node)

            if asizeof.asizeof(node_batch) >= 500000:
                try_pushing_nodes_to_queue(node_batch)
                node_batch = []

        if len(node_batch) > 0:
            try_pushing_nodes_to_queue(node_batch)
            node_batch = []

        for edge in edges:
            edge_batch.append(edge)

            if asizeof.asizeof(edge_batch) >= 500000:
                try_pushing_edges_to_queue(edge_batch)
                edge_batch = []

        if len(edge_batch) > 0:
            try_pushing_edges_to_queue(edge_batch)
            edge_batch = []

    # graph_data_deduplication.reset()


class GraphDataDeduplication:
    nodes_and_edges_map: dict

    def __init__(self):
        self.reset()

    def reset(self):
        self.nodes_and_edges_map = {}

    def deduplicate_nodes_and_edges(self, nodes: list, edges: list):
        final_nodes = []
        final_edges = []

        for node in nodes:
            node_key = str(node.id)
            if node_key not in self.nodes_and_edges_map:
                self.nodes_and_edges_map[node_key] = True
                final_nodes.append(node)

        for edge in edges:
            edge_key = str(edge[0]) + str(edge[2]) + str(edge[1])
            if edge_key not in self.nodes_and_edges_map:
                self.nodes_and_edges_map[edge_key] = True
                final_edges.append(edge)

        return final_nodes, final_edges


def try_pushing_nodes_to_queue(node_batch):
    try:
        save_data_points_queue.put((node_batch, []))
    except Exception:
        first_half, second_half = (
            node_batch[: len(node_batch) // 2],
            node_batch[len(node_batch) // 2 :],
        )
        save_data_points_queue.put((first_half, []))
        save_data_points_queue.put((second_half, []))


def try_pushing_edges_to_queue(edge_batch):
    try:
        save_data_points_queue.put(([], edge_batch))
    except Exception:
        first_half, second_half = (
            edge_batch[: len(edge_batch) // 2],
            edge_batch[len(edge_batch) // 2 :],
        )
        save_data_points_queue.put(([], first_half))
        save_data_points_queue.put(([], second_half))
