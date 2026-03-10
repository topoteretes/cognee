"""Skills client — nine methods for the full skill routing loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.models.node_set import NodeSet

from cognee.skills.execute import execute_skill
from cognee.skills.observe import record_skill_run
from cognee.skills.pipeline import ingest_skills, upsert_skills, remove_skill
from cognee.skills.promote import promote_skill_runs
from cognee.skills.retrieve import recommend_skills

logger = logging.getLogger(__name__)


class Skills:
    """Skill router with learned preferences.

    Usage::

        from cognee import skills

        await skills.ingest("./my_skills")
        recs = await skills.get_context("compress my conversation")
        full = await skills.load("summarize")
        result = await skills.execute("summarize", "compress this conversation")
        await skills.observe({"task_text": "...", "selected_skill_id": "summarize", "success_score": 0.9})
        await skills.promote()
    """

    async def ingest(
        self,
        skills_folder: Union[str, Path],
        dataset_name: str = "skills",
        source_repo: str = "",
        skip_enrichment: bool = False,
        node_set: str = "skills",
    ) -> None:
        """Parse SKILL.md files, enrich via LLM, and store in graph + vector index.

        Args:
            skills_folder: Path to directory containing skill subdirectories.
            dataset_name: Cognee dataset name to store skills under.
            source_repo: Provenance label (e.g. "my-org/skills").
            skip_enrichment: If True, skip LLM enrichment (parser output only).
            node_set: Tag for belongs_to_set (used for vector search scoping).
        """
        await ingest_skills(
            skills_folder=skills_folder,
            dataset_name=dataset_name,
            source_repo=source_repo,
            skip_enrichment=skip_enrichment,
            node_set=node_set,
        )

    async def upsert(
        self,
        skills_folder: Union[str, Path],
        dataset_name: str = "skills",
        source_repo: str = "",
        node_set: str = "skills",
    ) -> Dict[str, Any]:
        """Re-ingest skills, skipping unchanged, updating changed, removing deleted.

        Compares content hashes against existing graph nodes. Only changed or
        new skills go through LLM enrichment. Removed skills are deleted from
        both graph and vector stores.

        Returns a summary dict with unchanged/updated/added/removed counts.
        """
        return await upsert_skills(
            skills_folder=skills_folder,
            dataset_name=dataset_name,
            source_repo=source_repo,
            node_set=node_set,
        )

    async def remove(self, skill_id: str) -> bool:
        """Remove a single skill from the graph and vector stores.

        Returns True if the skill was found and deleted, False otherwise.
        """
        return await remove_skill(skill_id)

    async def get_context(
        self,
        task_text: str,
        top_k: int = 5,
        node_set: str = "skills",
    ) -> List[Dict[str, Any]]:
        """Return ranked skill recommendations for a task.

        Each entry includes a resolved task_pattern_id so it can be passed
        directly to observe().
        """
        recs = await recommend_skills(task_text, top_k=top_k, node_set=node_set)

        pattern_id = await self._resolve_pattern(task_text, recs)

        results = []
        for rec in recs:
            results.append(
                {
                    "skill_id": rec["skill_id"],
                    "name": rec["name"],
                    "score": rec["score"],
                    "vector_score": rec["vector_score"],
                    "prefers_score": rec["prefers_score"],
                    "instruction_summary": rec.get("instruction_summary", ""),
                    "task_pattern_id": pattern_id,
                    "tags": rec.get("tags", []),
                }
            )
        return results

    async def load(self, skill_id: str, node_set: str = "skills") -> Optional[Dict[str, Any]]:
        """Load full details for a skill by its skill_id."""
        engine = await get_graph_engine()
        raw_nodes, raw_edges = await engine.get_nodeset_subgraph(
            node_type=NodeSet, node_name=[node_set]
        )

        skill_node = None
        skill_nid = None
        for nid, props in raw_nodes:
            if props.get("type") == "Skill" and props.get("skill_id") == skill_id:
                skill_node = props
                skill_nid = str(nid)
                break

        if skill_node is None:
            return None

        patterns = []
        for src_id, tgt_id, rel_name, _ in raw_edges:
            if rel_name == "solves" and str(src_id) == skill_nid:
                for nid, props in raw_nodes:
                    if str(nid) == str(tgt_id) and props.get("type") == "TaskPattern":
                        patterns.append(
                            {
                                "pattern_key": props.get("pattern_key", ""),
                                "text": props.get("text", ""),
                                "category": props.get("category", ""),
                            }
                        )

        return {
            "skill_id": skill_node.get("skill_id", ""),
            "name": skill_node.get("name", ""),
            "instructions": skill_node.get("instructions", ""),
            "instruction_summary": skill_node.get("instruction_summary", ""),
            "description": skill_node.get("description", ""),
            "tags": skill_node.get("tags", []),
            "complexity": skill_node.get("complexity", ""),
            "source_path": skill_node.get("source_path", ""),
            "task_patterns": patterns,
        }

    async def execute(
        self,
        skill_id: str,
        task_text: str,
        context: Optional[str] = None,
        auto_observe: bool = True,
        session_id: str = "default",
        node_set: str = "skills",
    ) -> Dict[str, Any]:
        """Load a skill and execute it against a task via the configured LLM.

        Args:
            skill_id: The skill to execute.
            task_text: The user's task description.
            context: Optional additional context for the LLM.
            auto_observe: If True, automatically record the run to the observe cache.
            session_id: Session ID for the observation record.
            node_set: Graph node set to load the skill from.

        Returns:
            Dict with keys: output, skill_id, model, latency_ms, success, error.
        """
        skill = await self.load(skill_id, node_set=node_set)
        if skill is None:
            return {
                "output": "",
                "skill_id": skill_id,
                "model": "",
                "latency_ms": 0,
                "success": False,
                "error": f"Skill '{skill_id}' not found.",
            }

        result = await execute_skill(skill=skill, task_text=task_text, context=context)

        if auto_observe:
            await self.observe({
                "session_id": session_id,
                "task_text": task_text,
                "selected_skill_id": skill_id,
                "success_score": 1.0 if result["success"] else 0.0,
                "result_summary": result["output"][:500] if result["output"] else "",
                "latency_ms": result["latency_ms"],
                "error_type": "llm_error" if result["error"] else "",
                "error_message": result.get("error", "") or "",
            })

        return result

    async def list(self, node_set: str = "skills") -> List[Dict[str, Any]]:
        """List all ingested skills.

        Returns a list of dicts with skill_id, name, instruction_summary,
        tags, and complexity for each skill in the graph.
        """
        engine = await get_graph_engine()
        try:
            raw_nodes, _ = await engine.get_nodeset_subgraph(
                node_type=NodeSet, node_name=[node_set]
            )
        except Exception:
            return []

        results = []
        for _, props in raw_nodes:
            if props.get("type") == "Skill":
                results.append(
                    {
                        "skill_id": props.get("skill_id", ""),
                        "name": props.get("name", ""),
                        "instruction_summary": props.get("instruction_summary", ""),
                        "tags": props.get("tags", []),
                        "complexity": props.get("complexity", ""),
                    }
                )
        return results

    async def observe(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Record a skill execution to the short-term cache.

        Required keys in run: task_text, selected_skill_id, success_score.
        Optional: session_id, task_pattern_id, result_summary, candidate_skills,
                  feedback, error_type, error_message, latency_ms.
        """
        return await record_skill_run(
            session_id=run.get("session_id", "default"),
            task_text=run["task_text"],
            selected_skill_id=run["selected_skill_id"],
            task_pattern_id=run.get("task_pattern_id", ""),
            result_summary=run.get("result_summary", ""),
            success_score=run.get("success_score", 0.0),
            candidate_skills=run.get("candidate_skills"),
            feedback=run.get("feedback", 0.0),
            error_type=run.get("error_type", ""),
            error_message=run.get("error_message", ""),
            latency_ms=run.get("latency_ms", 0),
        )

    async def promote(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Promote cached runs to the long-term graph and update prefers edges."""
        return await promote_skill_runs(session_id=session_id)

    # ------------------------------------------------------------------

    async def _resolve_pattern(
        self, task_text: str, recs: List[Dict[str, Any]], node_set: str = "skills"
    ) -> str:
        """Resolve the best task_pattern_id for a query via vector search."""
        try:
            vector_engine = get_vector_engine()
            hits = await vector_engine.search(
                collection_name="TaskPattern_text",
                query_text=task_text,
                limit=1,
                include_payload=True,
            )
            if hits:
                hit_id = str(hits[0].id)
                engine = await get_graph_engine()
                raw_nodes, _ = await engine.get_nodeset_subgraph(
                    node_type=NodeSet, node_name=[node_set]
                )
                for nid, props in raw_nodes:
                    if str(nid) == hit_id and props.get("type") == "TaskPattern":
                        pk = props.get("pattern_key", "")
                        if pk:
                            return pk
        except Exception:
            pass

        if recs:
            patterns = recs[0].get("task_patterns", [])
            if patterns:
                return patterns[0].get("pattern_key", "")
        return ""


skills = Skills()
