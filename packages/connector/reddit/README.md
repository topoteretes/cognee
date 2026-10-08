# cognee-community-connector-reddit

A Reddit data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync subreddit conversations into memory — "ask your community".

It exposes a `dlt` source you hand to `cognee.remember(...)` / `cognee.add(...)`.
Submissions (each with a bounded comment tree) are rendered to text documents
and ingested as **normal documents** through cognee's cognify entity-extraction
pipeline (document-mode marker), not the deterministic dlt-row path.

## Requirements

Reddit closed its self-service public data API in November 2025. This connector
uses the still-supported path for developers: a **"script" OAuth app** plus the
OAuth 2.0 **password grant**. Create one at https://www.reddit.com/prefs/apps
(type *script*). Requires a cognee release with document-mode
(`DOCUMENT_SOURCE_ATTR` / `resolve_dlt_sources`).

## Install

```bash
uv pip install cognee-community-connector-reddit
# or, from this monorepo:
cd packages/connector/reddit && uv sync --all-extras
```

## Usage

```python
import cognee
from cognee_community_connector_reddit import reddit_source

await cognee.remember(
    reddit_source(
        subreddits=["machinelearning", "datascience"],
        sort="top",
        limit_per_subreddit=50,
        max_comments_per_post=80,  # comment-tree ceilings keep docs tidy
        max_comment_depth=5,
    ),
    dataset_name="reddit",
)

answer = await cognee.search(
    query_text="What libraries does the community recommend?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    datasets=["reddit"],
)
```

Credentials come from `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
`REDDIT_USERNAME`, `REDDIT_PASSWORD`, `REDDIT_USER_AGENT` (env), or the
`reddit_source(...)` kwargs. A descriptive `User-Agent` is required by Reddit.
See `examples/example.py`.

## Resource

| Resource | Notes |
|----------|-------|
| `reddit_posts` | One document per submission: title, allow-listed metadata, post body, and a bounded comment tree |

## Privacy & read-only guarantees

* Read-only — the connector never posts, votes, subscribes, or messages.
* Comment trees are depth- and count-bounded with explicit truncation markers.
* Documents only contain allow-listed fields (title, author, score, date,
  body, comments). No media URLs, removed-by flags, or hidden metadata.

## How sync + forget-on-delete work

Subreddit listings are cursor-paginated and each run is a **full snapshot**:
`write_disposition="replace"` rewrites staging with exactly the posts currently
in the listing. A post removed from the subreddit is absent from the new
snapshot, and cognee's `orphan_cleanup` forgets it from the graph and vector
stores on the next sync. Unchanged posts keep a stable content-hash `data_id`,
so only genuinely new/changed content is re-ingested. A fetch error aborts the
run instead of letting a partial snapshot forget live posts.

## Testing

```bash
uv run pytest tests/
```

All tests mock the Reddit API (no live credentials) and cover rendering,
whitespace collapse, comment-tree truncation, listing pagination, OAuth retry
(401 re-auth, 429 back-off), the env/credential guards, edit re-sync, and
forget-on-delete.