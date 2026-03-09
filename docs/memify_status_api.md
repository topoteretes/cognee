# Memify Status API

Cognee can start memify runs in background mode and return immediately. When that happens, the caller needs a stable way to check progress or completion later without guessing which run is the right one.

## Why `pipeline_run_id` exists

A dataset can be memified multiple times over its lifetime. Because of that, dataset-based status is only suitable for retrieving the latest memify run for a dataset. It is not a precise identifier for one specific asynchronous request.

`pipeline_run_id` solves that ambiguity:

- It identifies one concrete memify execution.
- It lets clients continue polling the same run they started earlier.
- It avoids mixing together older and newer memify runs for the same dataset.

## Start memify in background mode

```bash
curl -X POST https://your-cognee-host.example/api/v1/memify \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "research_notes",
    "run_in_background": true
  }'
```

Example response:

```json
{
  "status": "PipelineRunStarted",
  "pipeline_run_id": "8a8cc3be-c9de-4204-84de-f806686d0b4a",
  "dataset_id": "41069bd5-7292-4b8c-8d45-8169285affff",
  "dataset_name": "research_notes",
  "payload": []
}
```

## Query the exact memify run by `pipeline_run_id`

```bash
curl https://your-cognee-host.example/api/v1/memify/status/8a8cc3be-c9de-4204-84de-f806686d0b4a
```

Example response:

```json
{
  "pipeline_run_id": "8a8cc3be-c9de-4204-84de-f806686d0b4a",
  "dataset_id": "41069bd5-7292-4b8c-8d45-8169285affff",
  "pipeline_name": "memify_pipeline",
  "status": "DATASET_PROCESSING_COMPLETED",
  "created_at": "2025-01-01T00:00:00Z",
  "run_info": null,
  "dataset_name": "research_notes"
}
```

Persisted status records intentionally do not echo the raw memify input payload back through the status API. If a run errors, `run_info` may include concise error metadata without the original input data.

## Query the latest memify run for a dataset

By dataset name:

```bash
curl 'https://your-cognee-host.example/api/v1/memify/status?dataset_name=research_notes'
```

By dataset ID:

```bash
curl 'https://your-cognee-host.example/api/v1/memify/status?dataset_id=41069bd5-7292-4b8c-8d45-8169285affff'
```

If more than one readable dataset shares the same name, the dataset-name endpoint returns a conflict and the caller should retry with `dataset_id`.

Use dataset-based status for dashboards, summaries, or "latest run" views. Use `pipeline_run_id` when a client needs to track one exact asynchronous memify request.