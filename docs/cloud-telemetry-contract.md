# Cloud telemetry contract

What a cognee process sends as product telemetry when it runs in Cognee Cloud:
metadata about operations (what ran, how it went, how big it was), never customer
content. The policy behind it is CLO-799; the enforcement is
`cognee/shared/telemetry_schema.py`.

**Version:** 1 (`CLOUD_SCHEMA_VERSION`, sent on every event as `schema_version`).
Any change to the events or fields below bumps it.

## Selecting the profile

`TELEMETRY_ORIGIN=cloud` selects it. Cognee Cloud sets it for every tenant pod, in
the `cognee-tenant-operator` Helm charts in cloud-backend. Any other origin
(`sdk`, `api`, `cli`, `mcp`, `background`) keeps the full open-source payload,
unchanged. `TELEMETRY_DISABLED` turns telemetry off for both.

## Enforcement

Each event below has a strict Pydantic model (`CLOUD_EVENT_MODELS`):

- An event that is not listed is not sent.
- A field the event's model does not declare is removed; so is a field whose value
  fails its check (wrong type, not in its closed set, too long, too many items, too
  deeply nested). The rest of the event is still sent.
- Values are never coerced: `"5"` is not an int, `true` is not an int.
- Each removal is counted in `CLOUD_REJECTIONS` by event name and field path (for
  example `("cognee.recall", "unknown_field:datasets")`) and logged at debug level.
  The rejected value is never recorded.
- Any failure while applying the contract drops the event. It never changes the
  outcome of the operation that sent it.

## Envelope and trusted fields

```json
{
  "anonymous_id": "",
  "event_name": "cognee.recall",
  "event_time": "2026-11-09T15:36:54.948410+00:00",
  "user_properties": {"user_id": "…", "tenant_id": "…"},
  "properties": {"<event fields>": "…", "<trusted fields>": "…"}
}
```

`anonymous_id` is always `""`: the proxy requires the key, and a cloud event is
identified by user and tenant instead.

Trusted fields are built by `send_telemetry` itself and written last, so no event
property can replace them. They are validated too; if one fails, the event is not sent.

| Field | Values | Purpose |
|---|---|---|
| `user_id` | user UUID, or `sdk` / `unknown-user` / `None` when the caller has no user | who ran it |
| `tenant_id` | tenant UUID, or `Single User Tenant` | which tenant |
| `is_agent` | bool | the user is an agent (a child user with its own API key) |
| `agent_owner_id` | UUID or null | the human who owns the agent |
| `agent_plugin` | a known plugin key, `custom`, or null | which plugin the agent is |
| `schema_version` | int | this contract's version |
| `telemetry_origin` | `cloud` | profile marker |

## Event fields

Every event field is optional: call sites send different subsets. Every event may
carry `cognee_version` (string, ≤ 128 chars), the emitter's own value: abandoned-run
recovery sends `unknown`, since the run it reports may have died on another version.

**Pipeline run fields**

| Field | Values | Purpose |
|---|---|---|
| `pipeline_name` | UUID (the pipeline id, not its readable name) | groups runs of one pipeline |
| `pipeline_run_id` | UUID | correlates a run's events |
| `pipeline_event_scope` | `run` | run-level event, not item-level |
| `llm.provider` | an `LLMProvider` value | LLM backend |
| `llm.model` | string, ≤ 128 chars (the deployment's model alias) | LLM model |
| `llm.configured` | bool | an LLM is usable |
| `llm.structured_output` | `baml` / `litellm_native` / `instructor` | structured-output path |
| `llm.instructor_mode` | lowercase identifier, ≤ 32 chars | instructor mode in force |
| `embedding.provider`, `embedding.model` | string, ≤ 128 chars | embedding backend |
| `graph_extractor` | `llm` / `gliner_demo` / `unknown` / `invalid` | how the graph is extracted |
| `graph.provider`, `vector.provider`, `relational.provider` | string, ≤ 128 chars | database backends |

**Error fields**

| Field | Values | Purpose |
|---|---|---|
| `exception_type` | class name: identifier, ≤ 64 chars | error category |
| `exception_chain` | ≤ 4 class names, outermost first | which error wrapped which |
| `exception_cause` | class name | innermost error, when it differs from `exception_type` |
| `status_code` | int 100–599 | innermost HTTP status |

The exception class name is the error category. Messages, stack traces and local
variables are never sent.

**Events**

| Event | Fields |
|---|---|
| `Pipeline Run Started`, `Pipeline Run Completed` | pipeline run fields |
| `Pipeline Run Errored` | pipeline run fields, error fields, `recovered_at_startup` (bool: abandoned-run recovery) |
| `Pipeline Item Errored` | pipeline run fields, error fields, `item_count` (int), `item_loader` (loader registry name or `other`), `item_extension` (≤ 5 alphanumerics or `other`), `item_size_bucket`, `item_token_bucket` (closed size/token classes) |
| `Incremental Update Run Started` | `dataset_id`, `data_id` (UUID or `""`) |
| `Incremental Update Run Completed` | the above, `status` (`unchanged` / `incremental`), `regions`, `deleted_chunks`, `added_chunks`, `reused_chunks`, `redated_chunks`, `kept_chunks`, `reindexed_chunks`, `total_chunks` (ints) |
| `cognee.remember` | `mode` (`session` / `permanent`), `dataset_id`, `data_size_bytes`, `item_count`, `self_improvement`, `run_in_background` |
| `cognee.remember.code_graph` | `repository_count`, `index_vectors`, `run_in_background` |
| `cognee.remember.import` | `source_system` (a built-in importer or `cognee`), `mode` (an import mode), `run_in_background` |
| `cognee.recall` | `query_length`, `scope` (comma-joined recall scopes), `auto_route`, `top_k`, `search_type` (a `SearchType` value or `auto`), `dataset_ids` (≤ 100 comma-joined UUIDs), `include_references` |
| `cognee.recall ERRORED`, `cognee.search EXECUTION ERRORED` | error fields |
| `cognee.improve` | `session_count`, `run_in_background`, `build_global_context_index`, `build_truth_subspace`, `review_conflicts` |
| `cognee.forget` | `target` (what was forgotten: `everything`, `dataset`, `data_item`, …), `dataset_id`, `data_id` |
| `cognee.push` | `mode` (an import mode) |
| `cognee.export` | `format` (an export format) |
| `cognee.search EXECUTION STARTED` | `include_references` |
| `cognee.search EXECUTION COMPLETED` | trusted fields only |

## Removed from cloud events (migration)

What a cloud event carried before version 1 and no longer does:

| Removed | Use instead |
|---|---|
| `anonymous_id` (now `""`), `persistent_id` | `user_id`, `tenant_id` |
| `api_key_tracking_id`, `api_key_hash` | `user_id`, `tenant_id`; `agent_owner_id` for agents |
| `connected_tenant_id`, `connected_host` | `tenant_id` |
| `time` | `event_time` |
| `dataset_name`, `dataset`, `datasets` (fingerprints) | `dataset_id` / `dataset_ids` where the event has them |
| `session_id`, `session_ids` (fingerprints) | `session_count` (improve) |
| `graph.url`, `vector.url`, `relational.url` (fingerprints) | the `*.provider` fields |
| event `cognee.cognify DEFAULT TASKS CREATION ERRORED` | none: only the eval framework sent it |

`tenant_id` is still sent, now as a trusted field: an event's own copy is ignored.
`cognee_version` is still the event's own value.
