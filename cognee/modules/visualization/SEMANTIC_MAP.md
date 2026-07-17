# Semantic Memory Map

A **Semantic** tab in the graph visualization that lays the graph out by
*meaning* instead of topology. Each node is placed at the 2-D projection of its
embedding, so semantically similar nodes sit together and the graph's latent
structure — clusters of related entities — becomes visible at a glance.

It reuses vectors cognee already stores during `cognify`; nothing is
re-computed at query time and no embeddings are shipped to the browser beyond
2-D positions and precomputed neighbor lists.

## How it works

The production render path (`cognee_network_visualization`) builds the tab in
four steps, all bounded by one deterministic node sample
(`embedding_join.select_nodes`, cap 2000):

1. **`embedding_join.fetch_node_embeddings`** — joins graph nodes to their
   stored vectors. A graph node id is stored verbatim as its vector-row id
   (both sides use `str(data_point.id)`), so nodes join to the `{Type}_{field}`
   collections (`Entity_name`, `DocumentChunk_text`, …) directly. It issues one
   batched `retrieve(..., include_vector=True)` per collection — a LanceDB-only
   keyword; on other adapters the resulting `TypeError` falls back to
   re-embedding the indexed field (the stored vector is `embed(field)`, so the
   result matches, at the cost of embedding calls per render). A hit-rate log
   line makes a blank map diagnosable rather than silent.

2. **`layouts/semantic_layout.compute_positions`** — projects the embeddings to
   2-D with PCA (numpy SVD, sign-stabilized so the layout is deterministic).
   Positions are normalized, pinned, and rendered layout-once — no force
   simulation. Nodes without a vector are placed at their neighbor centroid, or
   on a ring if disconnected. Set `SEMANTIC_MAP_PROJECTION=umap` to use UMAP
   when `umap-learn` is installed (falls back to PCA when it isn't).

3. **`semantic_clusters.compute_clusters`** — clusters the *full-dimensional*
   embeddings with pure-numpy k-means (no scikit-learn) and precomputes each
   node's top-5 cosine neighbors for the hover panel. Clusters are labeled by
   their highest-degree usable entity names; `label_fn` is the seam for an LLM
   summarizer.

4. **Token substitution** — the orchestrator substitutes `__SEMANTIC_*__`
   tokens with the JS chunk and JSON payloads. The whole payload is best-effort:
   any failure (fetch, projection, clustering) logs a warning, the tokens become
   `null`, and the tab shows a friendly empty state — the classic render never
   breaks.

## Using it

The tab appears automatically in any `visualize_graph()` output:

```python
import cognee
from cognee.api.v1.visualize.visualize import visualize_graph

await cognee.add("...your text...")
await cognee.cognify()
await visualize_graph(destination_file_path="graph.html")
```

Open the HTML and click **Semantic** (or append `#semantic` to deep-link).
In the tab:

- **Cluster / Type** toggle recolors nodes by semantic cluster or ontology type.
- **Hover** a node to light up its nearest neighbors and list its relations.
- **Legend** entries filter to a single cluster or type; **zoom** with the
  scroll wheel or the on-screen controls.
- **Semantic ⇄ Structural** toggles between the pinned meaning-space layout and
  a bounded force layout over the graph topology.
- **Recall overlay** lights up the nodes a past recall query retrieved.

A runnable end-to-end example lives at
`examples/python/semantic_memory_map.py`.

## Design notes

- **Clustering runs on full-D embeddings**, not the 2-D projection, so groups
  reflect real structure rather than layout artifacts.
- **Determinism**: PCA sign convention, seeded k-means/sampling, and pinned
  positions make the layout reproducible — unit tests assert exact equality.
- **Zero new dependencies**: numpy (already core) does PCA and k-means; UMAP is
  an optional lazy import behind `SEMANTIC_MAP_PROJECTION=umap`.
