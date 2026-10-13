# Inspectable recall evidence

`POST /api/v1/recall` accepts optional `evidence_mode` (`evidenceMode`):

- `index`: returns an `EvidenceIndex` with at most 20 body-free items, an omitted
  reference count and an unaddressable-result count. This stays on the existing
  recall route so deployments retain their recall authorization/credit guards.
- `context`: keeps the recall list shape and attaches `metadata.evidence_items`
  and `metadata.evidence_contract = "recall_evidence_v1"` to graph results.
  Each artifact excerpt is capped at 1,400 characters, marked `truncated`, and
  retains the full artifact's version hash. Clients fit whole cited excerpts.

Both modes force `only_context` and `include_references`: they do not generate an
answer. JSON transport is required (streaming evidence requests return 422).
Omitting the mode preserves existing behavior. Health advertises
`recall_evidence_v1`.

References come from artifacts actually used by retrieval. Source passages are
prioritized before derived graph nodes/edges. This is context provenance, not a
claim that an artifact proves a particular sentence in an answer. Results without
existing graph evidence (including guidance/session/code results) remain
unaddressable; no synthetic source identity is fabricated for them.

## Batch expansion and versions

`POST /api/v1/recall/evidence` accepts `{"references": [...]}` with 1–5 typed
locators (`dataset_id`, `kind`, `artifact_id`, optional `version`; graph edges also
require their source/target node IDs and relationship name). It returns one
`EvidenceItem` per requested locator, in request order. No embedding or LLM call
is used. Nodes are fetched in one batch per authorized dataset; edge identities
and directed relationships are verified. Metadata includes the actual Data ID,
chunk index, document content hash and existing raw download route when available.
Raw downloads serve the current document, not an immutable historical copy.

`citation` is a resolvable `cognee://datasets/.../evidence/kind/id@version` locator.
Edge citations carry URL-encoded endpoint/relationship parameters. Clients may
parse the URI into a locator or pass the index's reference object directly.
The SHA-256 version covers the full projected artifact text, title and source
content/identity metadata, before excerpt limits. Access timestamps are excluded. It is not a Git revision or historical snapshot.

- `available`: the requested version still matches (or no version was supplied).
  Detail text is capped at 12,000 characters and reports truncation.
- `changed`: the requested version differs; `requested_version` records the old
  version and `reference`/`current_version` identify the current one. No replacement
  text is returned. Explicitly expand the current reference to inspect it.
- `unavailable`: artifact/source deleted, internal, or dataset inaccessible.
  These states deliberately share a result to avoid disclosing unauthorized data.
- `unsupported`: backend access control is disabled. This resolver declines shared
  graph reads because it cannot prove artifact membership for every graph kind.
- `error`: an authorized store failed. This is not a negative search result.

Every expansion rechecks dataset read grants and enters the dataset owner's
isolated graph context. Source chunks additionally require a live Data row in
that dataset. Arbitrary node properties and raw filesystem locations are never
returned. This feature needs no migration or source-body cache.

The Claude plugin's automatic graph recall requests context evidence in its
existing HTTP call, preserving its 45-second cloud deadline. Old servers ignore
the opt-in field and keep legacy recall; explicit index inspection reports
unsupported. Its local last-turn injection receipt contains IDs/hashes and
selection reasons, never source bodies or prompts.
