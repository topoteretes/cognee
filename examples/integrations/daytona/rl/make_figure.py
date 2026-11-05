"""Draw the reward grid of a run: every attempt as a square, per bug family and stage.

    python make_figure.py out/report-<run>.json results.svg

Rows are the four bug families plus a totals row; columns are the training
steps and the two held-out conditions. A filled square is an attempt whose
hidden tests all passed, an outlined one is a failure. The grid shows what a
pass-count table hides: which families flipped after which step, and that
groups come out uniform or mixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FAMILIES = [
    ("money_rounding", "money rounding"),
    ("idempotency", "idempotency keys"),
    ("retries", "retries"),
    ("refunds", "refunds"),
]

# One hue for the single state the chart is about; failures are outlines.
PASS = "#2a78d6"
FAIL_STROKE = "#b9b8b2"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
RULE = "#e6e5e0"
FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

CELL = 10  # square size
GAP = 3  # surface gap between squares
LABEL_W = 150
COL_W = 132
ROW_H = 46
TOP = 118
LEFT = 24


def columns(report: dict) -> list[dict]:
    """Column spec: title, caption, and the groups per family (variant -> rewards)."""
    cols = []
    for step in report["steps"]:
        n = step["step"]
        caption = {1: "empty memory", 2: "lessons from step 1", 3: "lessons from steps 1–2"}.get(
            n, f"after step {n - 1}"
        )
        cols.append(
            {
                "title": f"step {n}",
                "caption": caption,
                "groups": {g["family"]: g["rewards"] for g in step["groups"]},
            }
        )
    for condition, caption in (("memory off", "no lessons"), ("memory on", "lessons recalled")):
        data = report["evaluation"].get(condition)
        if data:
            cols.append(
                {
                    "title": f"held-out, {condition}",
                    "caption": caption,
                    "groups": {g["family"]: g["rewards"] for g in data["groups"]},
                    "heldout": True,
                }
            )
    return cols


def squares(x: float, y: float, rewards: list[int]) -> str:
    out = []
    for i, r in enumerate(rewards):
        sx = x + i * (CELL + GAP)
        if r:
            out.append(
                f'<rect x="{sx}" y="{y}" width="{CELL}" height="{CELL}" rx="2" fill="{PASS}"/>'
            )
        else:
            out.append(
                f'<rect x="{sx + 0.5}" y="{y + 0.5}" width="{CELL - 1}" height="{CELL - 1}" '
                f'rx="2" fill="none" stroke="{FAIL_STROKE}" stroke-width="1"/>'
            )
    return "".join(out)


def render(report: dict) -> str:
    cols = columns(report)
    n_train = sum(1 for c in cols if not c.get("heldout"))
    divider_x = LEFT + LABEL_W + n_train * COL_W - 14
    width = LEFT + LABEL_W + len(cols) * COL_W + 28 + 16
    height = TOP + (len(FAMILIES) + 1) * ROW_H + 40
    run = report.get("run", "")
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT}" font-size="12">'
        ),
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        (
            f'<text x="{LEFT}" y="30" font-size="16" font-weight="600" fill="{TEXT}">'
            "What memory changes, per bug family</text>"
        ),
        (
            f'<text x="{LEFT}" y="50" fill="{TEXT_2}">Each square is one attempt in its own sandbox; '
            "filled = every hidden test passed. Run "
            f"{run}, policy with no weight updates.</text>"
        ),
        # key
        f'<rect x="{LEFT}" y="64" width="{CELL}" height="{CELL}" rx="2" fill="{PASS}"/>',
        f'<text x="{LEFT + CELL + 6}" y="73" fill="{TEXT_2}">pass</text>',
        (
            f'<rect x="{LEFT + 48.5}" y="64.5" width="{CELL - 1}" height="{CELL - 1}" rx="2" '
            f'fill="none" stroke="{FAIL_STROKE}"/>'
        ),
        f'<text x="{LEFT + 48 + CELL + 6}" y="73" fill="{TEXT_2}">fail</text>',
        # section headers
        (
            f'<text x="{LEFT + LABEL_W}" y="{TOP - 28}" font-weight="600" fill="{TEXT}">'
            "training (8 attempts per group)</text>"
        ),
        (
            f'<text x="{LEFT + LABEL_W + n_train * COL_W}" y="{TOP - 28}" font-weight="600" '
            f'fill="{TEXT}">evaluation on unseen bugs (4 attempts)</text>'
        ),
        (
            f'<line x1="{divider_x}" y1="{TOP - 20}" x2="{divider_x}" '
            f'y2="{TOP + (len(FAMILIES) + 1) * ROW_H + 4}" stroke="{RULE}" stroke-width="1"/>'
        ),
    ]
    for ci, col in enumerate(cols):
        x = LEFT + LABEL_W + ci * COL_W
        parts.append(f'<text x="{x}" y="{TOP - 10}" fill="{TEXT}">{col["title"]}</text>')
        parts.append(
            f'<text x="{x}" y="{TOP + 4}" fill="{TEXT_2}" font-size="11">{col["caption"]}</text>'
        )

    y = TOP + 18
    totals = [0] * len(cols)
    sizes = [0] * len(cols)
    for key, label in FAMILIES:
        parts.append(
            f'<line x1="{LEFT}" y1="{y - 8}" x2="{width - 16}" y2="{y - 8}" stroke="{RULE}" '
            'stroke-width="1"/>'
        )
        parts.append(f'<text x="{LEFT}" y="{y + 15}" fill="{TEXT}">{label}</text>')
        for ci, col in enumerate(cols):
            rewards = col["groups"].get(key, [])
            x = LEFT + LABEL_W + ci * COL_W
            parts.append(squares(x, y + 6, rewards))
            totals[ci] += sum(rewards)
            sizes[ci] += len(rewards)
            if 0 < sum(rewards) < len(rewards):
                parts.append(
                    f'<text x="{x}" y="{y + 31}" fill="{TEXT_2}" font-size="10">mixed group</text>'
                )
        y += ROW_H
    # totals row
    parts.append(
        f'<line x1="{LEFT}" y1="{y - 8}" x2="{width - 16}" y2="{y - 8}" stroke="{TEXT_2}" '
        'stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{LEFT}" y="{y + 15}" font-weight="600" fill="{TEXT}">all families</text>'
    )
    for ci in range(len(cols)):
        x = LEFT + LABEL_W + ci * COL_W
        parts.append(
            f'<text x="{x}" y="{y + 15}" font-weight="600" fill="{TEXT}" '
            f'font-variant-numeric="tabular-nums">{totals[ci]}/{sizes[ci]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.write_text(render(report))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
