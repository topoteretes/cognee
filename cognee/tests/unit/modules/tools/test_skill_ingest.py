from __future__ import annotations

import asyncio
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4


_SKILL = """\
---
description: Review code changes for correctness and regressions.
allowed-tools: memory_search
---
# Instructions

Read the diff, identify concrete bugs, and cite file paths.
"""


def _run(coro):
    return asyncio.run(coro)


def _make_skill_dir(slug: str = "code-review") -> Path:
    root = Path(tempfile.mkdtemp(dir=Path.cwd()))
    skill_dir = root / slug
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_SKILL)
    return root


class TestSkillContract(unittest.TestCase):
    def setUp(self) -> None:
        # `set_database_global_context_variables` is a no-op when backend access
        # control is off; with default Kuzu+LanceDB it auto-enables and then tries
        # to look up the SimpleNamespace user in the relational DB. Disable it for
        # the duration of each test in this class.
        patcher = patch(
            "cognee.context_global_variables.backend_access_control_enabled",
            return_value=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_add_skills_persists_one_dataset_scoped_skill(self):
        from cognee.modules.engine.models import NodeSet
        from cognee.modules.tools.ingest_skills import add_skills

        root = _make_skill_dir()
        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        try:
            with patch(
                "cognee.modules.tools.ingest_skills.add_data_points",
                new_callable=AsyncMock,
            ) as mock_add:
                skills = _run(add_skills(root, user=user, dataset=dataset))

            assert len(skills) == 1
            skill = skills[0]
            assert skill.name == "code-review"
            assert skill.dataset_scope == [str(dataset.id)]
            assert skill.source_file.endswith("code-review/SKILL.md")
            assert skill.source_dir.endswith("code-review")
            assert isinstance(skill.belongs_to_set[0], NodeSet)
            assert skill.belongs_to_set[0].name == "skills"
            assert skill.content_hash
            assert "Review code changes" in skill.search_text
            assert skill.skill_text == skill.search_text
            assert mock_add.await_count == 1
            assert mock_add.await_args.kwargs["ctx"].dataset is dataset
        finally:
            import shutil

            shutil.rmtree(root)

    def test_same_skill_slug_in_two_datasets_gets_distinct_ids(self):
        from cognee.modules.tools.ingest_skills import add_skills

        root = _make_skill_dir()
        dataset_a = SimpleNamespace(id=uuid4(), name="a")
        dataset_b = SimpleNamespace(id=uuid4(), name="b")
        try:
            with patch(
                "cognee.modules.tools.ingest_skills.add_data_points",
                new_callable=AsyncMock,
            ):
                skill_a = _run(add_skills(root, dataset=dataset_a))[0]
                skill_b = _run(add_skills(root, dataset=dataset_b))[0]

            assert skill_a.name == skill_b.name
            assert skill_a.id != skill_b.id
            assert skill_a.dataset_scope == [str(dataset_a.id)]
            assert skill_b.dataset_scope == [str(dataset_b.id)]
        finally:
            import shutil

            shutil.rmtree(root)

    def test_agentic_completion_uses_agentic_retriever_only_when_explicit(self):
        from cognee.modules.retrieval.agentic_retriever import AgenticRetriever
        from cognee.modules.retrieval.graph_completion_cot_retriever import (
            GraphCompletionCotRetriever,
        )
        from cognee.modules.search.exceptions import UnsupportedSearchTypeError
        from cognee.modules.search.methods.get_search_type_retriever_instance import (
            get_search_type_retriever_instance,
        )
        from cognee.modules.search.types import SearchType

        user = SimpleNamespace(id=uuid4())
        dataset = SimpleNamespace(id=uuid4(), name="project")

        retriever = _run(
            get_search_type_retriever_instance(
                SearchType.AGENTIC_COMPLETION,
                "review this",
                user=user,
                dataset=dataset,
                retriever_specific_config={"skills": ["code-review"]},
            )
        )
        assert isinstance(retriever, AgenticRetriever)

        cot = _run(
            get_search_type_retriever_instance(
                SearchType.GRAPH_COMPLETION_COT,
                "review this",
                retriever_specific_config={},
            )
        )
        assert isinstance(cot, GraphCompletionCotRetriever)

        with self.assertRaises(UnsupportedSearchTypeError):
            _run(
                get_search_type_retriever_instance(
                    SearchType.GRAPH_COMPLETION,
                    "review this",
                    retriever_specific_config={"skills": ["code-review"]},
                )
            )

    def test_agentic_retriever_skips_memory_retrieval_when_graph_has_no_edges(self):
        from cognee.modules.retrieval.agentic_retriever import AgenticRetriever
        from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever

        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4())
        retriever = AgenticRetriever(skills=["code-review"], user=user, dataset=dataset)

        async def run_retrieval():
            with (
                patch.object(AgenticRetriever, "_graph_has_edges", new_callable=AsyncMock) as edges,
                patch(
                    "cognee.modules.retrieval.agentic_retriever.resolve_skills",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
                patch(
                    "cognee.modules.retrieval.agentic_retriever.list_tools_for_dataset",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
                patch.object(
                    GraphCompletionRetriever,
                    "get_retrieved_objects",
                    new_callable=AsyncMock,
                ) as parent_retrieval,
            ):
                edges.return_value = False
                result = await retriever.get_retrieved_objects(query="review this")
                return result, parent_retrieval

        result, parent_retrieval = _run(run_retrieval())

        assert result["triplets"] == []
        parent_retrieval.assert_not_awaited()

    def test_load_skill_loads_only_active_context_skills(self):
        from cognee.modules.engine.models import Skill
        from cognee.modules.tools.builtin.load_skill import handler
        from cognee.modules.tools.context import active_skills_var, opened_skills_var
        from cognee.modules.tools.errors import ToolInvocationError

        skill = Skill(name="code-review", description="Review code.", procedure="step 1")

        async def run_handler():
            active_token = active_skills_var.set({skill.name: skill})
            opened: set[str] = set()
            opened_token = opened_skills_var.set(opened)
            try:
                body = await handler({"name": "code-review"})
                with self.assertRaises(ToolInvocationError):
                    await handler({"name": "other"})
            finally:
                active_skills_var.reset(active_token)
                opened_skills_var.reset(opened_token)
            return body, opened

        body, opened = _run(run_handler())
        assert "step 1" in body
        assert opened == {"code-review"}

    def test_agentic_skill_run_write_is_dataset_scoped(self):
        from cognee.modules.engine.models import Skill
        from cognee.modules.engine.models.SkillRun import UNSCORED_SKILL_RUN_SCORE
        from cognee.modules.retrieval.agentic_retriever import AgenticRetriever

        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        skill = Skill(
            name="code-review",
            description="Review code.",
            skill_text="code-review\n\nReview code.",
            search_text="code-review\n\nReview code.",
            dataset_scope=[str(dataset.id)],
        )
        add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

        with patch.object(
            add_data_points_module, "add_data_points", new_callable=AsyncMock
        ) as mock_add:
            _run(
                AgenticRetriever._record_skill_runs(
                    [skill],
                    "review this",
                    "done",
                    user=user,
                    dataset=dataset,
                    session_id="session-1",
                )
            )

        run = mock_add.await_args.args[0][0]
        assert run.selected_skill_id == str(skill.id)
        assert run.selected_skill_name == "code-review"
        assert run.candidate_skills[0].skill_name == "code-review"
        assert run.candidate_skills[0].skill_description == "Review code."
        assert run.candidate_skills[0].skill_text == skill.skill_text
        assert run.candidate_skills[0].metadata["index_fields"] == ["skill_description"]
        assert run.dataset_scope == [str(dataset.id)]
        assert run.session_id == "session-1"
        assert run.success_score == UNSCORED_SKILL_RUN_SCORE
        assert mock_add.await_args.kwargs["ctx"].dataset is dataset

    def test_remember_skill_run_entry_persists_with_dataset_context(self):
        from cognee.memory import SkillRunEntry
        from cognee.modules.engine.models import Skill
        from cognee.modules.tools.skill_runs import remember_skill_run_entry

        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        skill = Skill(
            name="code-review",
            description="Review code.",
            procedure="step 1",
            skill_text="code-review\n\nReview code.\n\nstep 1",
            search_text="code-review\n\nReview code.\n\nstep 1",
            dataset_scope=[str(dataset.id)],
        )
        entry = SkillRunEntry(
            run_id="run-1",
            selected_skill_id="code-review",
            task_text="Review the diff",
            success_score=0.2,
            feedback=-0.5,
        )

        async def run_entry():
            with (
                patch("cognee.modules.tools.skill_runs.setup", new_callable=AsyncMock),
                patch(
                    "cognee.modules.tools.skill_runs.resolve_authorized_user_datasets",
                    new_callable=AsyncMock,
                    return_value=(user, [dataset]),
                ),
                patch(
                    "cognee.modules.tools.skill_runs.resolve_skills",
                    new_callable=AsyncMock,
                    return_value=[skill],
                ),
                patch(
                    "cognee.modules.tools.skill_runs.add_data_points",
                    new_callable=AsyncMock,
                ) as mock_add,
            ):
                run, resolved_dataset = await remember_skill_run_entry(
                    entry,
                    dataset_name="project",
                    session_id="session-1",
                    user=user,
                )
                return run, resolved_dataset, mock_add

        run, resolved_dataset, mock_add = _run(run_entry())
        assert resolved_dataset is dataset
        assert run.selected_skill_id == str(skill.id)
        assert run.selected_skill_name == "code-review"
        assert run.selected_skill is skill
        assert run.candidate_skills[0].skill_id == str(skill.id)
        assert run.candidate_skills[0].skill_name == "code-review"
        assert run.candidate_skills[0].skill_description == "Review code."
        assert "step 1" in run.candidate_skills[0].skill_text
        assert run.candidate_skills[0].metadata["index_fields"] == ["skill_description"]
        assert run.dataset_scope == [str(dataset.id)]
        assert run.belongs_to_set[0].name == "skills"
        assert mock_add.await_args.kwargs["ctx"].dataset is dataset

    def test_skill_improvement_proposal_does_not_mutate_skill(self):
        from cognee.modules.engine.models import Skill, SkillRun
        from cognee.modules.memify.skill_improvement import (
            SkillImprovementDraft,
            improve_skill,
        )

        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        skill = Skill(name="code-review", description="Review code.", procedure="old")
        run = SkillRun(
            run_id="run-1",
            session_id="session",
            task_text="review",
            selected_skill_id=str(skill.id),
            selected_skill_name=skill.name,
            dataset_scope=[str(dataset.id)],
            success_score=0.1,
        )

        async def create_proposal():
            with (
                patch(
                    "cognee.modules.memify.skill_improvement.find_skill_by_name",
                    new_callable=AsyncMock,
                    return_value=skill,
                ),
                patch(
                    "cognee.modules.memify.skill_improvement._find_recent_failure_runs",
                    new_callable=AsyncMock,
                    return_value=[run],
                ),
                patch(
                    "cognee.modules.memify.skill_improvement._generate_proposal",
                    new_callable=AsyncMock,
                    return_value=SkillImprovementDraft(
                        proposed_procedure="new",
                        rationale="The old procedure missed regression checks.",
                        confidence=0.8,
                    ),
                ),
                patch(
                    "cognee.modules.memify.skill_improvement.add_data_points",
                    new_callable=AsyncMock,
                ) as mock_add,
            ):
                proposal = await improve_skill("code-review", dataset=dataset, user=user)
                return proposal, mock_add

        proposal, mock_add = _run(create_proposal())
        assert skill.procedure == "old"
        assert proposal.old_procedure == "old"
        assert proposal.proposed_procedure == "# code-review\n\nnew"
        assert proposal.runs_used == ["run-1"]
        assert proposal.skill is skill
        assert proposal.runs == [run]
        assert proposal.dataset_scope == [str(dataset.id)]
        assert proposal.belongs_to_set[0].name == "skills"
        assert mock_add.await_args.args[0][0] is proposal

    def test_skill_improvement_apply_requires_existing_proposal_id(self):
        from cognee.modules.memify.skill_improvement import improve_skill

        dataset = SimpleNamespace(id=uuid4(), name="project")
        with self.assertRaisesRegex(ValueError, "proposal_id"):
            _run(improve_skill("code-review", dataset=dataset, apply=True))

    def test_skill_improvement_apply_updates_only_target_skill(self):
        from cognee.modules.engine.models import Skill, SkillImprovementProposal
        from cognee.modules.memify.skill_improvement import improve_skill

        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        skill = Skill(name="code-review", description="Review code.", procedure="old")
        proposal = SkillImprovementProposal(
            proposal_id="proposal-1",
            skill_id=str(skill.id),
            skill_name=skill.name,
            dataset_scope=[str(dataset.id)],
            old_procedure="old",
            proposed_procedure="new",
        )

        async def apply_proposal():
            with (
                patch(
                    "cognee.modules.memify.skill_improvement._find_proposal",
                    new_callable=AsyncMock,
                    return_value=proposal,
                ),
                patch(
                    "cognee.modules.memify.skill_improvement.find_skill_by_id",
                    new_callable=AsyncMock,
                    return_value=skill,
                ),
                patch(
                    "cognee.modules.memify.skill_improvement.add_data_points",
                    new_callable=AsyncMock,
                ) as mock_add,
            ):
                applied = await improve_skill(
                    "code-review",
                    dataset=dataset,
                    user=user,
                    proposal_id="proposal-1",
                    apply=True,
                )
                return applied, mock_add

        applied, mock_add = _run(apply_proposal())
        assert skill.procedure == "# code-review\n\nnew"
        assert skill.skill_text == "code-review\n\nReview code.\n\n# code-review\n\nnew"
        assert skill.search_text == skill.skill_text
        assert applied.status == "applied"
        assert mock_add.await_args.args[0] == [skill, proposal]

    def test_skill_improvement_apply_preserves_existing_skill_body_heading(self):
        from cognee.modules.engine.models import Skill, SkillImprovementProposal
        from cognee.modules.memify.skill_improvement import improve_skill

        dataset = SimpleNamespace(id=uuid4(), name="project")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        skill = Skill(name="code-review", description="Review code.", procedure="old")
        proposal = SkillImprovementProposal(
            proposal_id="proposal-1",
            skill_id=str(skill.id),
            skill_name=skill.name,
            dataset_scope=[str(dataset.id)],
            old_procedure="old",
            proposed_procedure="# code-review\n\n- Read the diff.",
        )

        async def apply_proposal():
            with (
                patch(
                    "cognee.modules.memify.skill_improvement._find_proposal",
                    new_callable=AsyncMock,
                    return_value=proposal,
                ),
                patch(
                    "cognee.modules.memify.skill_improvement.find_skill_by_id",
                    new_callable=AsyncMock,
                    return_value=skill,
                ),
                patch(
                    "cognee.modules.memify.skill_improvement.add_data_points",
                    new_callable=AsyncMock,
                ),
            ):
                return await improve_skill(
                    "code-review",
                    dataset=dataset,
                    user=user,
                    proposal_id="proposal-1",
                    apply=True,
                )

        _run(apply_proposal())
        assert skill.procedure == "# code-review\n\n- Read the diff."
