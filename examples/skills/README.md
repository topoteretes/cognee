# Skills examples

These examples are plain `SKILL.md` folders that can be ingested with
`cognee.remember`.

```python
import cognee

await cognee.remember(
    "examples/skills",
    dataset_name="hackathon",
    node_set=["skills"],
    enrich=False,
)
```

The Daytona hackathon demo embeds the same `code-review` skill so sandbox
runs work without copying files from this directory.

Agents can report skill quality through the same `remember` surface:

```python
from cognee.memory import SkillRunEntry

await cognee.remember(
    SkillRunEntry(
        selected_skill_id="code-review",
        task_text="Review the authentication changes",
        result_summary="Found one missing permission check...",
        success_score=0.25,
        feedback=-0.5,
        error_type="missing_permission_check",
        error_message="The review missed a dataset ownership path.",
    ),
    dataset_name="hackathon",
    session_id="agent-review",
    improve=True,
    improve_min_runs=1,
)
```
