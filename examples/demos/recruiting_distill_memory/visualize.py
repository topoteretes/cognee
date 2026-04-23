"""Render the human_memory graph (seed + agent-approved rules) as HTML.

Runs inside a custom pipeline so the per-user ACL scope is active —
same reason inspect_rulebook.py and review_pending_rules.py do this.
Opens the output in the default browser when done.
"""

import asyncio
import webbrowser
from pathlib import Path
from typing import Any

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user


HERE = Path(__file__).parent
OUTPUT_HTML = HERE / "output" / "human_memory_graph.html"


async def _render(_data: Any, ctx: PipelineContext = None) -> str:
    return await visualize_graph(str(OUTPUT_HTML))


async def main() -> None:
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    user = await get_default_user()
    await cognee.run_custom_pipeline(
        tasks=[Task(_render)],
        data=[None],
        dataset="human_memory",
        user=user,
        pipeline_name="visualize_human_memory",
    )
    print(f"\nWrote {OUTPUT_HTML}")
    print("Opening in browser ...")
    webbrowser.open(f"file://{OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
