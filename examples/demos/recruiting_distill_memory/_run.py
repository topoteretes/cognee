"""Shared runner: call the three decorated tools and assemble an interview plan.

Imported by run_naive.py / run_grounded.py *after* they set the env flags
(RECRUITING_WITH_MEMORY, RECRUITING_SESSION_ID) — this module's import of
`agent_tools` reads those flags at its own import time.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

from examples.demos.recruiting_distill_memory.agent_tools import (
    SESSION_ID,
    WITH_MEMORY,
    compose_screen_invite,
    format_candidate,
    propose_interview_format,
    schedule_panel,
)

HERE = Path(__file__).parent
CANDIDATE_PATH = HERE / "data" / "candidates" / "dev_rao.json"
OUTPUT_DIR = HERE / "output"


async def run_plan(output_filename: str) -> dict[str, Any]:
    candidate = json.loads(CANDIDATE_PATH.read_text())
    summary = format_candidate(candidate)

    mode = "grounded" if WITH_MEMORY else "naive"
    print(f"=== Run mode: {mode}  (session_id={SESSION_ID}) ===")
    print(f"Candidate: {candidate['name']} ({candidate['prior_company']} → {candidate['role']})\n")

    print("1/3 propose_interview_format ...")
    fmt = await propose_interview_format(candidate_summary=summary)
    print(f"    format={fmt.format}  duration={fmt.duration_minutes}m  applied={fmt.applied_rule_ids}")

    print("2/3 schedule_panel ...")
    panel = await schedule_panel(candidate_summary=summary)
    print(
        f"    panelists={panel.panelists}  total_hours={panel.total_hours}  "
        f"cto_included={panel.cto_included}  applied={panel.applied_rule_ids}"
    )

    print("3/3 compose_screen_invite ...")
    invite = await compose_screen_invite(candidate_summary=summary)
    print(f"    subject={invite.subject!r}  applied={invite.applied_rule_ids}")

    plan = {
        "mode": mode,
        "session_id": SESSION_ID,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate": candidate,
        "interview_format": fmt.model_dump(),
        "panel": panel.model_dump(),
        "screen_invite": invite.model_dump(),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / output_filename
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"\nWrote {out_path.relative_to(HERE.parent.parent.parent)}")
    return plan
