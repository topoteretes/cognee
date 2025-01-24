import os
import cognee
import importlib.util

# from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP
from cognee.api.v1.search import SearchType
from cognee.shared.data_models import KnowledgeGraph

mcp = FastMCP("cognee", timeout=120000)


@mcp.tool()
async def cognify(text: str, graph_model_file: str = None, graph_model_name: str = None) -> str:
    """Build knowledge graph from the input text"""
    if graph_model_file and graph_model_name:
        graph_model = load_class(graph_model_file, graph_model_name)
    else:
        graph_model = KnowledgeGraph

    await cognee.add(text)

    try:
        await cognee.cognify(graph_model=graph_model)
    except Exception as e:
        raise ValueError(f"Failed to cognify: {str(e)}")

    return "Ingested"


@mcp.tool()
async def search(search_query: str) -> str:
    """Search the knowledge graph"""
    search_results = await cognee.search(SearchType.INSIGHTS, query_text=search_query)

    results = retrieved_edges_to_string(search_results)

    return results


@mcp.tool()
async def prune() -> str:
    """Reset the knowledge graph"""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    return "Pruned"


# @mcp.tool()
# async def visualize() -> Image:
#     """Visualize the knowledge graph"""
#     try:
#         image_path = await cognee.visualize_graph()

#         img = PILImage.open(image_path)
#         return Image(data=img.tobytes(), format="png")
#     except (FileNotFoundError, IOError, ValueError) as e:
#       raise ValueError(f"Failed to create visualization: {str(e)}")


def node_to_string(node):
    node_data = ", ".join(
        [f'{key}: "{value}"' for key, value in node.items() if key in ["id", "name"]]
    )

    return f"Node({node_data})"


def retrieved_edges_to_string(search_results):
    edge_strings = []
    for triplet in search_results:
        node1, edge, node2 = triplet
        relationship_type = edge["relationship_name"]
        edge_str = f"{node_to_string(node1)} {relationship_type} {node_to_string(node2)}"
        edge_strings.append(edge_str)
    return "\n".join(edge_strings)


def load_class(model_file, model_name):
    model_file = os.path.abspath(model_file)
    spec = importlib.util.spec_from_file_location("graph_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, model_name)

    return model_class


# def get_freshest_png(directory: str) -> Image:
#     if not os.path.exists(directory):
#         raise FileNotFoundError(f"Directory {directory} does not exist")

#     # List all files in 'directory' that end with .png
#     files = [f for f in os.listdir(directory) if f.endswith(".png")]
#     if not files:
#         raise FileNotFoundError("No PNG files found in the given directory.")

#     # Sort by integer value of the filename (minus the '.png')
#     # Example filename: 1673185134.png -> integer 1673185134
#     try:
#         files_sorted = sorted(files, key=lambda x: int(x.replace(".png", "")))
#     except ValueError as e:
#         raise ValueError("Invalid PNG filename format. Expected timestamp format.") from e

#     # The "freshest" file has the largest timestamp
#     freshest_filename = files_sorted[-1]
#     freshest_path = os.path.join(directory, freshest_filename)

#     # Open the image with PIL and return the PIL Image object
#     try:
#         return PILImage.open(freshest_path)
#     except (IOError, OSError) as e:
#         raise IOError(f"Failed to open PNG file {freshest_path}") from e

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
