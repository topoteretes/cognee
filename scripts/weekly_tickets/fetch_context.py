"""Build the weekly digest that both analysis variants consume.

Pulls the last 7 days of docs-assistant conversations from MotherDuck
(analytics.analytics.mintlify_chatbot_conversations), buckets them by theme,
extracts error-flavored reports, and (optionally) lists currently open Linear
tickets so the analyzer can avoid proposing duplicates.

Requires: MOTHERDUCK_TOKEN. Optional: LINEAR_API_KEY.
Output: digest.md in the current directory.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import duckdb

THEMES = {
    "docker / deployment": ["docker", "deploy", "kubernetes", "helm", "container"],
    "local / self-hosted setup": [
        "local",
        "self-host",
        "install",
        "setup",
        "quick start",
        "getting started",
    ],
    "llm / model config": [
        "llm",
        "ollama",
        "openai",
        "model",
        "api key",
        "gemini",
        "anthropic",
        "azure",
        "embedding",
    ],
    "search / retrieval": ["search", "retriev", "query", "rag"],
    "datasets / data mgmt": ["dataset", "delete", "prune", "forget"],
    "graph / ontology": ["graph", "ontolog", "entity", "entities", "node", "edge"],
    "backend databases": [
        "neo4j",
        "postgres",
        "pgvector",
        "qdrant",
        "lancedb",
        "kuzu",
        "database",
        "sqlite",
    ],
    "mcp / agents": ["mcp", "claude", "agent", "cursor", "copilot"],
    "memory / sessions": ["memory", "remember", "session", "recall"],
    "pipelines / ingestion": ["cognify", "pipeline", "ingest", "chunk", "upload"],
    "errors / not working": [
        "error",
        "fail",
        "not work",
        "stuck",
        "exception",
        "traceback",
        "401",
        "404",
        "422",
        "429",
        "500",
    ],
    "pricing / cloud / auth": ["pricing", "cost", "cloud", "token", "auth", "login"],
}

ERROR_MARKERS = [
    "error",
    "fail",
    "not work",
    "stuck",
    "doesn't",
    "problem",
    "traceback",
    "exception",
    "401",
    "404",
    "422",
    "429",
    "500",
]


def fetch_docs_digest(con, since):
    total = con.execute(
        "SELECT count(*) FROM analytics.analytics.mintlify_chatbot_conversations WHERE created_at >= ?",
        [since],
    ).fetchone()[0]

    titles = [
        r[0]
        for r in con.execute(
            """SELECT title FROM analytics.analytics.mintlify_chatbot_conversations
               WHERE created_at >= ? AND title IS NOT NULL""",
            [since],
        ).fetchall()
    ]

    theme_counts = {
        theme: sum(1 for t in titles if any(k in t.lower() for k in keywords))
        for theme, keywords in THEMES.items()
    }

    error_reports = [t[:400] for t in titles if any(m in t.lower() for m in ERROR_MARKERS)][:60]

    return total, theme_counts, error_reports


def fetch_open_linear_titles():
    """Open SDK/COG issue titles, for dedup context. Best-effort."""
    api_key = os.getenv("LINEAR_API_KEY")
    if not api_key:
        return None
    import urllib.request

    query = {
        "query": """query { issues(first: 200, filter: {state: {type: {nin: ["completed","canceled"]}},
                    team: {key: {in: ["SDK","COG"]}}}) { nodes { identifier title } } }"""
    }
    req = urllib.request.Request(
        "https://api.linear.app/graphql",
        data=json.dumps(query).encode(),
        headers={"Authorization": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            nodes = json.load(resp)["data"]["issues"]["nodes"]
        return [f"{n['identifier']}: {n['title']}" for n in nodes]
    except (OSError, ValueError, KeyError, TypeError) as e:
        # Best effort: the digest is still useful without the dedup list.
        # OSError covers urllib.error.URLError/HTTPError and socket timeouts;
        # ValueError covers malformed JSON; KeyError/TypeError an unexpected
        # (e.g. GraphQL "errors"-only) response shape.
        print(f"warning: could not fetch Linear issues: {e}", file=sys.stderr)
        return None


def main():
    if not os.getenv("MOTHERDUCK_TOKEN"):
        sys.exit("MOTHERDUCK_TOKEN is required")
    os.environ["motherduck_token"] = os.environ["MOTHERDUCK_TOKEN"]

    since = datetime.now(timezone.utc) - timedelta(days=7)
    con = duckdb.connect("md:")
    total, theme_counts, error_reports = fetch_docs_digest(con, since)
    open_tickets = fetch_open_linear_titles()

    lines = [
        f"# Docs-assistant digest — week ending {datetime.now(timezone.utc).date()}",
        f"\nConversations in the last 7 days: **{total}**\n",
        "## Question volume by theme\n",
    ]
    for theme, n in sorted(theme_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {theme}: {n}")

    lines.append("\n## Error / problem reports (raw user text, truncated)\n")
    for t in error_reports:
        lines.append(f"- {t!r}")

    if open_tickets:
        lines.append("\n## Already-open Linear tickets (do NOT propose duplicates)\n")
        lines.extend(f"- {t}" for t in open_tickets)

    with open("digest.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"digest.md written: {total} conversations, {len(error_reports)} error reports")


if __name__ == "__main__":
    main()
