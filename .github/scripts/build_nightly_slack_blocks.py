"""Build the nightly Slack message's `blocks` array.

Every performance arm renders the same way: a title line carrying the job
result and run count, one row per metric with p50/p90/p99, and a link to the
HTML report. Expressing that 14 times in YAML meant 14 copies of the same six
lines, each repeating `fromJSON(needs.X.outputs.Y || '{}')` twenty-odd times —
which is how the Rust arms came to render differently from the Python ones in
the first place (CLO-488). Adding an arm is now one line in the `ARMS` env var.

Input (env):
  ARMS         one arm per line: `emoji|title|result|metrics_json|url`.
               Blank lines are ignored. `metrics_json` is a perf job's
               `metrics` output; empty/unparseable is treated as `{}`, which is
               what a failed job produces. A `tenant_create` key adds the
               cloud-only tenant row.
  STATUS_*     header fields (emoji, summary, ran_at, branch, cadence,
               sha — the last three fall back if unset).
  RUN_URL      link target for the footer.

Output: `blocks=<compact JSON>` appended to $GITHUB_OUTPUT (stdout if unset).
JSON is a subset of YAML, so the workflow interpolates the result straight into
the action's YAML `payload`.
"""

import json
import os
import sys

# Rows in render order. The second element is the padding between the label and
# `p50`, preserved verbatim from the hand-written YAML this replaced: the
# columns do not line up (add/cognify/tenant end at 9, search/total at 11), and
# reproducing that exactly is what lets the migration be diffed to zero.
ROWS_BEFORE_SEARCH = [("add", 6), ("cognify", 2)]
ROWS_AFTER_SEARCH = [("total", 6)]
TENANT_ROW = ("tenant", 3)
# Search rows, in render order. The Python and cloud arms time each search type
# separately (`search_graph` / `search_hybrid`); the Rust SDK arm still reports
# a single `search`. search_rows() picks whichever the arm actually carries.
SEARCH_ROWS = [("search", 5), ("search graph", 3), ("search hybrid", 2)]
# Metric rows read `<key>.p50` etc. from the arm's metrics object; these are the
# rows whose label differs from its JSON key.
METRIC_KEY = {
    "tenant": "tenant_create",
    "search graph": "search_graph",
    "search hybrid": "search_hybrid",
}
PERCENTILES = ("p50", "p90", "p99")


def section(text):
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def search_rows(metrics):
    """The search rows this arm reports.

    A failed job produces `{}`, which matches none of them — fall back to the
    single legacy row so the arm still renders a search line with blank
    numbers rather than silently losing it.
    """
    rows = [row for row in SEARCH_ROWS if METRIC_KEY.get(row[0], row[0]) in metrics]
    return rows or [SEARCH_ROWS[0]]


def metric_row(metrics, label, pad):
    stats = metrics.get(METRIC_KEY.get(label, label)) or {}
    cells = "  •  ".join(f"{p} `{stats.get(p, '')}s`" for p in PERCENTILES)
    return f"{label}{' ' * pad}{cells}"


def arm_block(emoji, title, result, metrics_json, url):
    try:
        metrics = json.loads(metrics_json) if metrics_json.strip() else {}
    except json.JSONDecodeError:
        # A failed job leaves the output empty or partial. Render the arm with
        # blank numbers rather than dropping it — a missing section reads as
        # "this suite does not exist", which is worse than a visibly empty one.
        metrics = {}

    rows = [TENANT_ROW] if "tenant_create" in metrics else []
    rows += ROWS_BEFORE_SEARCH + search_rows(metrics) + ROWS_AFTER_SEARCH

    lines = [f"*{emoji} {title}* (`{result}`)  •  runs `{metrics.get('success', '')}`"]
    lines += [metric_row(metrics, label, pad) for label, pad in rows]
    lines.append(f"<{url}|HTML report>")
    return section("\n".join(lines) + "\n")


def main():
    env = os.environ
    # Fall back rather than raise: a hand-dispatch or an older caller may not
    # set these, and a missing label must not cost the whole message.
    branch = env.get("STATUS_BRANCH") or "?"
    cadence = env.get("STATUS_CADENCE") or "manual"
    sha = (env.get("STATUS_SHA") or "")[:7]
    header = section(
        f"{env.get('STATUS_EMOJI', '')} *Nightly Tests* — `{branch}` — "
        f"{env.get('STATUS_SUMMARY', '')}\n"
        f"*Ran at:* `{env.get('STATUS_RAN_AT', '')}`  •  "
        f"*Branch:* `{branch}` (`{cadence}`)  •  *Commit:* `{sha}`\n"
    )

    blocks = [header]
    for line in env.get("ARMS", "").splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 5:
            raise SystemExit(
                f"ARMS line must have 5 pipe-separated fields, got {len(fields)}: {line!r}"
            )
        blocks.append(arm_block(*(f.strip() for f in fields)))

    blocks.append(section(f"<{env.get('RUN_URL', '')}|View run>\n"))

    payload = f"blocks={json.dumps(blocks, ensure_ascii=False, separators=(',', ':'))}"
    out = env.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
