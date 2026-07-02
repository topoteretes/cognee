# 5-Minute Onboarding: Notetaker Bot

This guide explains how to use the Notetaker Bot API to ingest meeting transcripts and query them with temporal awareness.

## 1. Start the API Server

Ensure your Cognee API is running:
```bash
uv run python -m cognee.api.client
```

## 2. Ingest a Meeting Transcript

Send a `POST` request to `/api/v1/notetaker/ingest`. 

The payload requires a `series_id` (the dataset grouping all occurrences of this meeting) and the transcript broken down into speaker turns.

```bash
curl -X POST "http://localhost:8000/api/v1/notetaker/ingest" \
-H "Content-Type: application/json" \
-d '{
  "series_id": "weekly_standup",
  "meeting_id": "standup_june_24",
  "permalink": "https://slack.com/archives/C123/p456",
  "turns": [
    ["Alice", "Deploy is blocked on staging creds.", "2026-06-24 14:32"],
    ["Bob", "I will fix the staging creds today.", "2026-06-24 14:33"]
  ]
}'
```

This endpoint returns `202 Accepted` and immediately delegates cognify to a background worker. The response
includes `data_ids` (the ingested occurrence's Data UUIDs — keep these to forget a single meeting later) and a
`pipeline_run_id`.

## 3. Track Processing Status

Use the dataset status endpoint to check if the background pipeline has finished cognifying:

```bash
curl -X GET "http://localhost:8000/api/v1/datasets/status"
```

## 4. Recall Action Items or Decisions

Once the pipeline is complete, query the temporal graph using the recall endpoint. Recall is **scoped to the
series** via `series_id`, so temporal "what changed" spans that series' occurrences without leaking across
other meetings:

```bash
curl -X GET "http://localhost:8000/api/v1/notetaker/recall?series_id=weekly_standup&query=What%20are%20the%20action%20items?&query_type=action_items"
```

*Supported `query_type` values:*
- `action_items`
- `decisions`
- `temporal_delta` (e.g., for "what changed since last week?")

Results are grounded in the citation prefix `[Speaker, Date, permalink=...]` injected during ingestion.

## 5. Forget Data

Delete an entire series, or a single occurrence by the `data_id` returned from `/ingest`:

```bash
# Forget the entire series (dataset)
curl -X POST "http://localhost:8000/api/v1/notetaker/forget" \
-H "Content-Type: application/json" \
-d '{"series_id": "weekly_standup"}'

# Forget a single occurrence within the series
curl -X POST "http://localhost:8000/api/v1/notetaker/forget" \
-H "Content-Type: application/json" \
-d '{"series_id": "weekly_standup", "data_id": "<data_id-from-ingest>"}'
```
