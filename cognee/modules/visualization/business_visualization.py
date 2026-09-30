"""Business visualization — the UI's canvas renderer as a standalone page.

Same recipe as ``cognee_network_visualization``: the graph is preprocessed on
the server, the result is embedded as JSON into an HTML shell together with a
JS chunk, and the browser that opens the page does the rendering. Nothing
executes JS here. The JS chunk is the bundled Business canvas from
``cognee-frontend`` (see ``views/business_standalone.py``), so the page shows
the same visualization as the UI and needs no network access.
"""

import os

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.modules.visualization.cognee_network_visualization import (
    _safe_json_embed,
    build_visualization_payload,
)
from cognee.modules.visualization.views import business_standalone
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "business_template.html")


def build_business_visualization_html(
    graph_data,
    dataset_name: str = "",
    search_events: list | None = None,
) -> str:
    """The Business page as a string: payload JSON plus the renderer bundle."""
    payload = build_visualization_payload(graph_data, search_events=search_events)
    payload["dataset_name"] = dataset_name

    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("__DATASET_NAME__", dataset_name)
    html = html.replace("__BUSINESS_JS__", business_standalone.emit_js())
    # Payload last, so a payload string can never be mistaken for a token.
    return html.replace("__PAYLOAD__", _safe_json_embed(payload))


async def cognee_business_visualization(
    graph_data,
    destination_file_path: str | None = None,
    dataset_name: str = "",
    search_events: list | None = None,
) -> str:
    """Render the Business page to ``destination_file_path`` and return the HTML."""
    html = build_business_visualization_html(
        graph_data, dataset_name=dataset_name, search_events=search_events
    )

    if not destination_file_path:
        destination_file_path = os.path.join(os.path.expanduser("~"), "graph_visualization.html")

    file_storage = LocalFileStorage(os.path.dirname(destination_file_path))
    await file_storage.store(os.path.basename(destination_file_path), html, overwrite=True)

    logger.info(f"Graph visualization saved as {destination_file_path}")

    return html
