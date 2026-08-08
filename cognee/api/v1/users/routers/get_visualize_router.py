from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from uuid import UUID
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.shared.logging_utils import get_logger
from cognee.modules.visualization.subgraph_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_SEED_TOP_K,
)
from cognee.modules.users.methods import get_authenticated_user, get_user
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.models import User

from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


class UserDatasetPair(BaseModel):
    user_id: UUID = Field(
        description=(
            "UUID of the user who owns the dataset "
            "(superuser-only endpoint; obtain via the permissions/users APIs)."
        ),
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    dataset_id: UUID = Field(
        description=(
            "UUID of the dataset to include in the combined visualization "
            "(must be readable by user_id)."
        ),
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
    )


def get_visualize_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=None)
    async def visualize(
        dataset_id: UUID = Query(
            ...,
            description=(
                "UUID of the dataset to visualize. "
                "List your datasets via GET /api/v1/datasets to find it."
            ),
            examples=[""],
        ),
        full: bool = Query(
            False,
            description="Render the entire graph instead of a bounded subgraph.",
        ),
        query: Optional[str] = Query(
            None,
            description="Query string whose nearest vector hits seed the subgraph.",
        ),
        seed_node_ids: Optional[List[str]] = Query(
            None,
            description="Explicit seed node ids for subgraph neighborhood expansion.",
        ),
        neighborhood_depth: int = Query(
            DEFAULT_NEIGHBORHOOD_DEPTH,
            ge=1,
            le=10,
            description="k-hop neighborhood depth for subgraph expansion.",
        ),
        neighborhood_seed_top_k: int = Query(
            DEFAULT_SEED_TOP_K,
            ge=1,
            le=100,
            description="Maximum number of seed nodes.",
        ),
        max_nodes: int = Query(
            DEFAULT_MAX_NODES,
            ge=1,
            le=5000,
            description="Hard cap on rendered nodes after expansion.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Generate an HTML visualization of the dataset's knowledge graph.

        By default renders a bounded subgraph around relevant seed nodes; pass
        ``full=true`` to render the entire graph (legacy behavior). Seeds come
        from ``query`` or ``seed_node_ids`` when given, otherwise the graph's
        highest-degree nodes.

        ## Query Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset to visualize
        - **full** (bool): Render the full graph when true
        - **query** (str): Query string to seed the subgraph via vector search
        - **seed_node_ids** (list[str]): Explicit seed node ids
        - **neighborhood_depth** (int): k-hop expansion depth (default 2)
        - **neighborhood_seed_top_k** (int): Max seeds (default 10)
        - **max_nodes** (int): Node cap after expansion (default 500)

        ## Response
        Returns an HTML page containing the interactive graph visualization.

        ## Error Codes
        - **409 Conflict**: Dataset not found, permission denied, or visualization
          failed (detail in the `error` field)

        ## Notes
        - User must have read permissions on the dataset
        - Visualization is interactive and allows graph exploration
        """
        send_telemetry(
            "Visualize API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/visualize",
                "dataset_id": str(dataset_id),
                "full": full,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_graph

        try:
            # Verify user has permission to read dataset
            dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

            html_visualization = await visualize_graph(
                dataset=dataset[0].id,
                user=user,
                full=full,
                query=query,
                seed_node_ids=seed_node_ids,
                neighborhood_depth=neighborhood_depth,
                neighborhood_seed_top_k=neighborhood_seed_top_k,
                max_nodes=max_nodes,
            )
            return HTMLResponse(html_visualization)

        except Exception as error:
            logger.exception("Visualization failed for dataset %s", dataset_id)
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.get("/json", response_model=None)
    async def visualize_json(
        dataset_id: UUID = Query(
            ...,
            description=(
                "UUID of the dataset to visualize. "
                "List your datasets via GET /api/v1/datasets to find it."
            ),
            examples=[""],
        ),
        full: bool = Query(
            False,
            description="Include the entire graph instead of a bounded subgraph.",
        ),
        query: Optional[str] = Query(
            None,
            description="Query string whose nearest vector hits seed the subgraph.",
        ),
        seed_node_ids: Optional[List[str]] = Query(
            None,
            description="Explicit seed node ids for subgraph neighborhood expansion.",
        ),
        neighborhood_depth: int = Query(
            DEFAULT_NEIGHBORHOOD_DEPTH,
            ge=1,
            le=10,
            description="k-hop neighborhood depth for subgraph expansion.",
        ),
        neighborhood_seed_top_k: int = Query(
            DEFAULT_SEED_TOP_K,
            ge=1,
            le=100,
            description="Maximum number of seed nodes.",
        ),
        max_nodes: int = Query(
            DEFAULT_MAX_NODES,
            ge=1,
            le=5000,
            description="Hard cap on rendered nodes after expansion.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Return the dataset's knowledge graph as a JSON-safe payload.

        Same authorization, dataset resolution and bounded fetch as `GET
        /visualize` — pass the same arguments to get the JSON behind the
        same subgraph the HTML page would have shown. Does not include the
        semantic layout; see `GET /visualize/semantic` for that, computed
        separately so a client that never opens the semantic tab never pays
        for it.

        ## Query Parameters
        Same as `GET /visualize` (dataset_id, full, query, seed_node_ids,
        neighborhood_depth, neighborhood_seed_top_k, max_nodes).

        ## Response
        A JSON object with `nodes`, `links`, `color_maps`, `schema_graph`,
        `schema_data`, `pipeline_stages`, `edge_classes`, `bundles`,
        `provenance_index`, `has_meaningful_topological_rank`, `memory_map`
        and `search_events`.

        ## Error Codes
        - **409 Conflict**: Dataset not found, permission denied, or the
          payload could not be built (generic message; full detail is
          server-logged, not returned, to avoid leaking internals)

        ## Notes
        - User must have read permissions on the dataset
        """
        send_telemetry(
            "Visualize JSON API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/visualize/json",
                "dataset_id": str(dataset_id),
                "full": full,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_graph_json

        try:
            dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

            payload = await visualize_graph_json(
                dataset=dataset[0].id,
                user=user,
                full=full,
                query=query,
                seed_node_ids=seed_node_ids,
                neighborhood_depth=neighborhood_depth,
                neighborhood_seed_top_k=neighborhood_seed_top_k,
                max_nodes=max_nodes,
            )
            return JSONResponse(status_code=200, content=payload)

        except Exception:
            logger.exception("Visualization JSON payload failed for dataset %s", dataset_id)
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build the visualization payload"},
            )

    @router.get("/semantic", response_model=None)
    async def visualize_semantic(
        dataset_id: UUID = Query(
            ...,
            description=(
                "UUID of the dataset to visualize. "
                "List your datasets via GET /api/v1/datasets to find it."
            ),
            examples=[""],
        ),
        full: bool = Query(
            False,
            description="Include the entire graph instead of a bounded subgraph.",
        ),
        query: Optional[str] = Query(
            None,
            description="Query string whose nearest vector hits seed the subgraph.",
        ),
        seed_node_ids: Optional[List[str]] = Query(
            None,
            description="Explicit seed node ids for subgraph neighborhood expansion.",
        ),
        neighborhood_depth: int = Query(
            DEFAULT_NEIGHBORHOOD_DEPTH,
            ge=1,
            le=10,
            description="k-hop neighborhood depth for subgraph expansion.",
        ),
        neighborhood_seed_top_k: int = Query(
            DEFAULT_SEED_TOP_K,
            ge=1,
            le=100,
            description="Maximum number of seed nodes.",
        ),
        max_nodes: int = Query(
            DEFAULT_MAX_NODES,
            ge=1,
            le=5000,
            description="Hard cap on rendered nodes after expansion.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Return semantic positions and clusters for the same subgraph as `GET /visualize/json`.

        Pass the same arguments used for `GET /visualize/json` to lay out the
        same subgraph semantically. This is the one call that fetches
        embeddings (up to 2000 nodes) and runs PCA/UMAP over them, so it is
        only worth calling when the semantic tab is actually open.

        ## Query Parameters
        Same as `GET /visualize/json`.

        ## Response
        A JSON object with `semantic_positions` and `semantic_clusters`,
        either of which is `null` when there are no embeddings to lay out.

        ## Error Codes
        - **409 Conflict**: Dataset not found, permission denied, or the
          payload could not be built (generic message; full detail is
          server-logged, not returned, to avoid leaking internals)

        ## Notes
        - User must have read permissions on the dataset
        """
        send_telemetry(
            "Visualize Semantic API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/visualize/semantic",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_semantic_json

        try:
            dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

            payload = await visualize_semantic_json(
                dataset=dataset[0].id,
                user=user,
                full=full,
                query=query,
                seed_node_ids=seed_node_ids,
                neighborhood_depth=neighborhood_depth,
                neighborhood_seed_top_k=neighborhood_seed_top_k,
                max_nodes=max_nodes,
            )
            return JSONResponse(status_code=200, content=payload)

        except Exception:
            logger.exception("Semantic payload failed for dataset %s", dataset_id)
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build the semantic layout"},
            )

    @router.get("/brains", response_model=None)
    async def visualize_brains(
        max_nodes: int = Query(
            DEFAULT_MAX_NODES,
            ge=1,
            le=5000,
            description="Hard cap on rendered nodes per dataset.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Return every dataset the caller may read, each as a small graph preview.

        No `dataset_id`: this is the Brains overview, not one brain. Built on
        the same union CLO-399 already uses to decide which datasets a user
        can see — their own, their tenant's, and anything granted to a role
        they belong to — so a dataset shared with a group appears here with
        no separate authorization logic.

        ## Query Parameters
        - **max_nodes** (int): Node cap applied independently to each
          dataset (default 500) — there is no larger, separate cap for "all
          datasets at once".

        ## Response
        `{dataset_id: {"name", "nodes", "links", "node_set_colors"}}`.

        ## Notes
        - Only datasets the caller has read permission on are included
        """
        send_telemetry(
            "Visualize Brains API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/visualize/brains",
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import build_brains_payload

        try:
            payload = await build_brains_payload(user=user, max_nodes=max_nodes)
            return JSONResponse(status_code=200, content=payload)

        except Exception:
            logger.exception("Brains overview payload failed")
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build the brains overview"},
            )

    @router.get("/live-events", response_model=None)
    async def visualize_live_events(
        dataset_id: UUID = Query(
            ...,
            description=(
                "UUID of the dataset this poll is for. Gates who may call this "
                "endpoint (same read-permission check as every other visualize "
                "route) — the events themselves are the caller's own, not "
                "filtered to this dataset's graph."
            ),
            examples=[""],
        ),
        since: Optional[datetime] = Query(
            None,
            description=(
                "Cursor from a previous call's response. Omit on the first "
                "call to get every available event."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Return search/improve events newer than a cursor, for the Memory tab's live timeline.

        Meant to be polled instead of re-fetching the whole `GET
        /visualize/json` payload just to refresh the timeline: pass the
        previous response's `cursor` back as `since` and only new events
        come back. The filter is strict, so nothing is ever delivered twice.

        ## Query Parameters
        - **dataset_id** (UUID): authorization only, see above
        - **since** (datetime, optional): cursor from a previous call

        ## Response
        `{"events": [...], "cursor": <ISO datetime or null>}`

        ## Error Codes
        - **403 Forbidden**: Caller lacks read permission on the dataset (or
          it does not exist)
        - **409 Conflict**: Payload could not be built (generic message;
          full detail is server-logged, not returned, to avoid leaking
          internals)

        ## Notes
        - User must have read permissions on the dataset
        """
        send_telemetry(
            "Visualize Live Events API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/visualize/live-events",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import get_live_events

        try:
            payload = await get_live_events(dataset_id, since=since, user=user)
            return JSONResponse(status_code=200, content=payload)

        except PermissionDeniedError:
            return JSONResponse(
                status_code=403,
                content={"error": "Not authorized to read this dataset"},
            )
        except Exception:
            logger.exception("Live events payload failed for dataset %s", dataset_id)
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to fetch live events"},
            )

    @router.post("/multi", response_model=None)
    async def visualize_multi(
        pairs: List[UserDatasetPair],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Generate a combined HTML visualization of graph data from multiple users' datasets.

        This endpoint aggregates knowledge graphs from multiple user+dataset pairs
        into a single interactive visualization, with each user's nodes tagged for
        color-by-user rendering.

        ## Request Body
        A JSON array of objects, each with:
        - **user_id** (UUID): The user who owns the dataset
        - **dataset_id** (UUID): The dataset to include

        ## Response
        Returns an HTML page containing the combined interactive graph visualization.

        ## Error Codes
        - **403 Forbidden**: Caller is not a superuser
        - **409 Conflict**: A user/dataset pair does not exist, is not readable,
          or visualization failed (detail in the `error` field)

        ## Notes
        - Requires superuser privileges to view other users' data
        - Each user+dataset pair must exist and be accessible
        """
        send_telemetry(
            "Visualize Multi API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/visualize/multi",
                "pair_count": len(pairs),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_multi_user_graph

        try:
            if not user.is_superuser:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Superuser privileges required for multi-user visualization"},
                )

            user_dataset_pairs = []
            for pair in pairs:
                target_user = await get_user(pair.user_id)
                datasets = await get_authorized_existing_datasets(
                    [pair.dataset_id], "read", target_user
                )
                user_dataset_pairs.append((target_user, datasets[0]))

            html_visualization = await visualize_multi_user_graph(user_dataset_pairs)
            return HTMLResponse(html_visualization)

        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
