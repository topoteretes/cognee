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

This endpoint returns `202 Accepted` and immediately delegates processing to a background worker.

## 3. Track Processing Status

You will receive a `pipeline_info` object in the ingest response. Use the dataset status endpoint to check if the background pipeline has finished cognifying:

```bash
curl -X GET "http://localhost:8000/api/v1/datasets/status"
```

## 4. Recall Action Items or Decisions

Once the pipeline is complete, query the temporal graph using the recall endpoint:

```bash
curl -X GET "http://localhost:8000/api/v1/notetaker/recall?query=What%20are%20the%20action%20items?&query_type=action_items"
```

*Supported `query_type` values:*
- `action_items`
- `decisions`
- `temporal_delta` (e.g., for "what changed since last week?")

The response will include the exact citation prefix `[Speaker, Date, permalink=...]` injected during ingestion.

## 5. Forget Data

If you need to delete a specific occurrence or an entire meeting series:

```bash
# Forget the entire series
curl -X POST "http://localhost:8000/api/v1/notetaker/forget" \
-H "Content-Type: application/json" \
-d '{"series_id": "weekly_standup"}'
```
