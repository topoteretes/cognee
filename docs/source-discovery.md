# Discover and route stored memory sources

Source discovery derives an authorized catalog from existing dataset and document
metadata. It adds no tables, memory copies, provider enumeration, connector fetches
or UI requirements. A source target is either a dataset or a node set in a dataset.
The same node-set name in different datasets has distinct, stable target IDs.

Ingestion should attach meaningful `node_set` names, document `label`, and optional
`external_metadata.source` and `external_metadata.source_description` strings.
These fields describe provenance and topics; they do not grant access. Catalog
routing sends only these projected fields to the configured Cognee LLM, not raw
text, arbitrary metadata or ambient session memory. Labels can contain private
information, so the configured LLM must be appropriate for the deployment.

## HTTP contract

All routes use the existing authenticated principal and native dataset read ACLs.

- `GET /api/v1/datasets/source-catalog`: optional repeated `dataset_ids`, `offset`
  and `limit` (1–200). Returns descriptors, total, next offset and completeness.
- `POST /api/v1/datasets/source-route`: `query`, optional free-form `source_hint`,
  `dataset_ids`, `exclude_source_ids`, `max_sources` (1–8, default 6), and
  `max_catalog_entries` (1–2048, default 512). Returns validated targets and reasons.
  Routing does not search content. Explicit inaccessible datasets fail the request.
- `GET /api/v1/datasets/source-documents/{source_id}`: metadata pages, `limit`
  (1–500), and optional `after` document UUID. Membership and permissions are
  rechecked for each page. `next_cursor: null` ends the stored-document listing.
- `GET /api/v1/datasets/source-document/{dataset_id}/{document_id}`: authorized
  metadata for a document, including canonical IDs and node sets. Native legacy
  document aliases resolve within the selected dataset.

Search chosen targets with existing `POST /api/v1/search`, using `dataset_ids`,
`node_name`, and `node_name_filter_operator: "AND" | "OR"`. Both HTTP search and
recall forward this operator. Read originals using the existing raw-data route.

Catalog capabilities describe stored-document operations only: `search`,
`list_documents`, `read_document`. They do not advertise live connector access or
SQL execution. An imported schema is searchable memory; live database questions
still use the existing authorized Cognee tool connection.

Named-source resolution first matches exact catalog names or homogeneous document
provenance. A mixed topic group cannot substitute for an exact named source. The
LLM receives local integer handles; server code maps selected handles back to
authorized targets. Excluding all matches returns an inconclusive route rather
than broadening to other sources. Free-form hints without an exact metadata match
use semantic selection and are reported as such.

## Budgets and coverage

The catalog projects at most 50,000 native document metadata rows per request. It
reports `complete: false` if this bound is exceeded. Offset pages are not snapshots;
concurrent changes may move catalog entries between pages. Document pages use UUID
keysets, not event timestamps, and also do not provide snapshot isolation.

Routing considers descriptors in batches of 64, with up to three concurrent LLM
calls per request. Batch winners are reduced and jointly ranked until at most eight
remain. The response reports the number of LLM calls. This is heuristic selection,
not an exhaustive search or a benchmarked recall guarantee. Broad questions may
need a second search excluding previous targets. Never translate an inconclusive
route, budget error or empty ranked result into proof that information is absent.

A catalog larger than the explicit routing budget returns `catalog_budget_exceeded`
without calling the LLM. Narrow the dataset selection or deliberately increase the
budget. No catalog descriptor is silently dropped to fit a prompt. An incomplete
metadata scan is still reported as incomplete even if routing selects targets.

Descriptions and labels are samples, not generated summaries of all content. Sparse
metadata can make routing inconclusive. Improving ingestion metadata improves
routing without adding provider-specific plugin logic. This first implementation
scans current native metadata; very large archives would benefit from a native
incremental catalog index rather than a larger per-request scan limit.
