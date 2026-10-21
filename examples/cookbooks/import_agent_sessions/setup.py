"""Write sample Claude Code and Codex sessions, so the demo runs without any history of your own.

Writes sample/claude/projects/... and sample/codex/sessions/... in the same JSON-lines formats
the two agents write to ~/.claude and ~/.codex, tool calls and injected instructions included,
so the demo runs the same readers your own history goes through. The sessions are two weeks
of work on a made-up web shop, dated relative to today. No key is needed to write them.

Run: uv run python examples/cookbooks/import_agent_sessions/setup.py
"""

import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"
CWD = "/Users/alex/code/acme-shop"

CLAUDE_SESSIONS = [
    {
        "id": "7f3c2a10-5b4e-4c1d-9a8f-2e6b1d0c4a01",
        "days_ago": 12,
        "branch": "fix/checkout-double-charge",
        "title": "Fix double charge on checkout retry",
        "turns": [
            (
                (
                    "Customers get charged twice when they click Pay again after a timeout. "
                    "Can you find out why? The checkout code is in src/checkout/."
                ),
                (
                    "The retry path in src/checkout/pay.ts creates a new Stripe PaymentIntent on "
                    "every click, so a click after a timeout starts a second payment while the first "
                    "one still completes. I made the order id the Stripe idempotency key, so a retry "
                    "returns the existing PaymentIntent instead of creating one, and disabled the Pay "
                    "button while a request is in flight. The test in pay.test.ts now clicks twice "
                    "and asserts one charge."
                ),
            ),
            (
                "Good. Does the webhook handle the duplicate too?",
                (
                    "Yes, after one change: payment_intent.succeeded could arrive twice for the same "
                    "order, and the handler marked the order paid and sent the receipt each time. It "
                    "now checks orders.paid_at first and returns early when it is set, so the receipt "
                    "email goes out once."
                ),
            ),
        ],
    },
    {
        "id": "9a1d4e77-2c3b-4f60-8e15-7b9c0a2d3e02",
        "days_ago": 4,
        "branch": "feat/order-history-page",
        "title": "Add the order history page",
        "turns": [
            (
                "Add an order history page under /account/orders. Newest first, 20 per page.",
                (
                    "Added app/account/orders/page.tsx with a server component that loads the user's "
                    "orders newest first, 20 per page with cursor pagination on created_at. Each row "
                    "links to the existing order detail page. I added an index on "
                    "orders(user_id, created_at) in migration 0042, since the query scanned the table."
                ),
            ),
            (
                "Show the refund status as well.",
                (
                    "Each row now shows a Refunded or Partly refunded badge, read from the refunds "
                    "table that the Stripe charge.refunded webhook fills in."
                ),
            ),
        ],
    },
]

CODEX_SESSIONS = [
    {
        "id": "01a2b3c4-d5e6-7f80-9a1b-2c3d4e5f6a03",
        "days_ago": 9,
        "branch": "feat/api-rate-limit",
        "turns": [
            (
                "Add rate limiting to the public API so one client can't flood /api/products.",
                (
                    "Added a token-bucket limiter in src/api/middleware/rate_limit.ts backed by "
                    "Redis: 60 requests a minute per API key, or per IP without one. Over the limit "
                    "the API answers 429 with a Retry-After header. It is applied to every /api route "
                    "except /api/webhooks, so Stripe's webhook calls are never limited."
                ),
            ),
        ],
    },
    {
        "id": "01a2b3c4-d5e6-7f80-9a1b-2c3d4e5f6a04",
        "days_ago": 2,
        "branch": "fix/stock-oversell",
        "turns": [
            (
                (
                    "Two orders bought the last item at the same time and we oversold. Fix the stock "
                    "check in checkout."
                ),
                (
                    "The stock check read products.stock and decremented it in two statements, so two "
                    "checkouts could both see 1 in stock. Checkout now decrements in one statement, "
                    "UPDATE products SET stock = stock - 1 WHERE id = $1 AND stock > 0, and fails the "
                    "order with an out-of-stock message when no row is updated. A test runs two "
                    "checkouts for the last item concurrently and asserts one fails."
                ),
            ),
        ],
    },
]


def at(days_ago: int, minutes: int = 0) -> str:
    moment = datetime.now().astimezone() - timedelta(days=days_ago) + timedelta(minutes=minutes)
    return moment.isoformat()


def write_lines(path: Path, records: list[dict], days_ago: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    stamp = time.time() - days_ago * 86_400  # the file was last written when the session ran,
    os.utime(path, (stamp, stamp))  # so the reader doesn't take it for a session still running


def claude_records(session: dict) -> list[dict]:
    base = {
        "sessionId": session["id"],
        "cwd": CWD,
        "gitBranch": session["branch"],
        "entrypoint": "cli",
        "isSidechain": False,
    }
    records = []
    for number, (prompt, answer) in enumerate(session["turns"]):
        stamp = at(session["days_ago"], number * 10)
        records += [
            # what you typed, with the reminder Claude Code adds to it
            {
                **base,
                "type": "user",
                "timestamp": stamp,
                "message": {
                    "role": "user",
                    "content": f"<system-reminder>Today's date is {stamp[:10]}.</system-reminder>"
                    + prompt,
                },
            },
            # a tool call and its result: both are dropped
            {
                **base,
                "type": "assistant",
                "timestamp": stamp,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"t{number}", "name": "Grep", "input": {}}
                    ],
                },
            },
            {
                **base,
                "type": "user",
                "timestamp": stamp,
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"t{number}", "content": "..."}
                    ],
                },
            },
            {
                **base,
                "type": "assistant",
                "timestamp": stamp,
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            },
        ]
    records.append({"type": "ai-title", "sessionId": session["id"], "aiTitle": session["title"]})
    return records


def codex_records(session: dict) -> list[dict]:
    started = at(session["days_ago"])
    records = [
        {
            "type": "session_meta",
            "timestamp": started,
            "payload": {
                "id": session["id"],
                "timestamp": started,
                "cwd": CWD,
                "originator": "codex_cli_rs",
                "git": {"branch": session["branch"]},
            },
        }
    ]
    for prompt, answer in session["turns"]:
        records += [
            # the instructions Codex injects into the first message are dropped
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"# AGENTS.md instructions for {CWD}"},
                        {"type": "input_text", "text": prompt},
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "I'll look at the code first."}],
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "shell", "input": "rg stock"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": answer}],
                },
            },
        ]
    return records


def write_sample() -> None:
    """Write every sample session, replacing an earlier sample/ folder."""
    shutil.rmtree(SAMPLE, ignore_errors=True)
    project = SAMPLE / "claude" / "projects" / CWD.replace("/", "-")
    for session in CLAUDE_SESSIONS:
        write_lines(
            project / f"{session['id']}.jsonl", claude_records(session), session["days_ago"]
        )
    for session in CODEX_SESSIONS:
        day = datetime.now().astimezone() - timedelta(days=session["days_ago"])
        path = (
            SAMPLE
            / "codex"
            / "sessions"
            / f"{day:%Y/%m/%d}"
            / f"rollout-{day:%Y-%m-%dT%H-%M-%S}-{session['id']}.jsonl"
        )
        write_lines(path, codex_records(session), session["days_ago"])


if __name__ == "__main__":
    write_sample()
    print(
        f"[setup] Wrote {len(CLAUDE_SESSIONS)} Claude Code and {len(CODEX_SESSIONS)} Codex"
        " sample sessions to sample/."
    )
    print(
        "[setup] Now run: uv run python examples/cookbooks/import_agent_sessions/import_agent_sessions.py"
    )
