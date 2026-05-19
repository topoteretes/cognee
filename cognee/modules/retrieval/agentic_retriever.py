"""Retriever that unifies memory retrieval, skills, and tool calls.

Extends GraphCompletionRetriever so memory triplet retrieval and context
formatting are reused. Adds a ReAct-style loop: at each turn the LLM emits a
structured AgentStep (either a tool_call or a final_answer). Tool calls flow
through execute_tool, which enforces per-dataset permissions the same way
search does.

Skill procedure bodies are not loaded into the system prompt up-front. Only
their names and descriptions are shown in the catalog; the LLM calls the
load_skill tool to fetch a body on demand (progressive disclosure).
"""

import time
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Union
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, Field

from cognee.modules.engine.models import Skill, Tool
from cognee.modules.engine.models.SkillRun import ToolCall as SkillRunToolCall
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.modules.tools.context import active_skills_var, opened_skills_var
from cognee.modules.tools.errors import (
    ToolInvocationError,
    ToolPermissionError,
    ToolScopeError,
)
from cognee.modules.tools.execute_tool import execute_tool
from cognee.modules.tools.registry import list_tools_for_dataset
from cognee.modules.tools.resolve_skills import resolve_skills
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger


logger = get_logger("AgenticRetriever")
MAX_TOOL_OUTPUT_CHARS = 8_000


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call; must be in the manifest.")
    arguments: dict = Field(
        default_factory=dict,
        description="Arguments for the tool, matching its input_schema.",
    )


class AgentStep(BaseModel):
    """One LLM turn: either request a tool call or emit the final answer."""

    thought: str = Field(
        default="",
        description="Short reasoning for the chosen action.",
    )
    tool_call: Optional[ToolCall] = Field(
        default=None,
        description="Populate when another tool call is needed; else leave null.",
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Populate when you have enough to answer; else leave null.",
    )


class AgenticRetriever(GraphCompletionRetriever):
    """
    Memory + skills + tools in one retriever.

    Instance attributes beyond GraphCompletionRetriever's:
    - explicit_skills: skills or skill names supplied by the caller.
    - tool_filter: optional whitelist of tool names to expose this turn.
    - user: the user whose permissions gate every tool call.
    - dataset_id: the dataset under which memory and tools are scoped.
    - max_iter: upper bound on tool-call iterations before forcing a final answer.
    - agentic_system_prompt_path, agentic_user_prompt_path: prompt templates
      used inside the loop (distinct from the parent's graph-completion prompts).
    """

    def __init__(
        self,
        skills: Optional[Sequence[Union[str, Skill]]] = None,
        tools: Optional[List[str]] = None,
        user: Optional[User] = None,
        dataset=None,
        dataset_id: Optional[UUID] = None,
        max_iter: int = 6,
        agentic_system_prompt_path: str = "agentic_system.txt",
        agentic_user_prompt_path: str = "agentic_user.txt",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if user is None:
            raise ValueError("AgenticRetriever requires `user` for ACL-checked tool execution.")
        self.explicit_skills = list(skills) if skills else []
        self.tool_filter = tools
        self.user = user
        self.dataset = dataset
        self.dataset_id = dataset_id or getattr(dataset, "id", None)
        if self.dataset_id is None:
            raise ValueError("AgenticRetriever requires one explicit dataset for skill lookup.")
        self.max_iter = max_iter
        self.agentic_system_prompt_path = agentic_system_prompt_path
        self.agentic_user_prompt_path = agentic_user_prompt_path
        self._cached_context: Optional[str] = None

    def _use_session_cache(self) -> bool:
        """Use the explicit retriever user instead of relying only on ContextVar state."""
        from cognee.infrastructure.databases.cache.config import CacheConfig

        return bool(getattr(self.user, "id", None) and CacheConfig().caching)

    async def get_retrieved_objects(  # type: ignore[override]
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> Any:
        """Return a dict with memory triplets, active skills, and permitted tools.

        The return type widens the parent's List[Edge] return because the downstream
        methods on this class consume a richer bundle — the search pipeline passes
        this object through verbatim to get_context_from_objects.
        """
        skills = await resolve_skills(
            self.explicit_skills,
            dataset_id=self.dataset_id,
        )
        tools = await self._resolve_active_tools(skills)
        triplets = []

        if await self._graph_has_edges():
            triplets = await super().get_retrieved_objects(query=query, query_batch=query_batch)
        else:
            validate_retriever_input(query, query_batch, self._use_session_cache())

        return {"triplets": triplets, "skills": skills, "tools": tools}

    async def _graph_has_edges(self) -> bool:
        """A skills-only graph has no memory triplets to retrieve."""
        from cognee.infrastructure.databases.graph import get_graph_engine

        try:
            graph_engine = await get_graph_engine()
            _, edges = await graph_engine.get_graph_data()
        except Exception as exc:
            logger.warning("Unable to inspect graph edges before agentic retrieval: %s", exc)
            return True
        return bool(edges)

    async def _resolve_active_tools(self, skills: List[Skill]) -> List[Tool]:
        """Ambient tools intersected with skill-declared tools (union across skills)."""
        all_tools = await list_tools_for_dataset(dataset_id=self.dataset_id)

        if self.tool_filter is not None:
            allowed_names = set(self.tool_filter)
            all_tools = [t for t in all_tools if t.name in allowed_names]

        declared: set[str] = {"load_skill"} if skills else set()
        for skill in skills:
            declared.update(skill.declared_tools or [])

        if skills:
            all_tools = [t for t in all_tools if t.name in declared]

        return all_tools

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects=None,
    ) -> Union[str, List[str]]:
        if not isinstance(retrieved_objects, dict):
            return await super().get_context_from_objects(
                query=query, query_batch=query_batch, retrieved_objects=retrieved_objects
            )

        triplets = retrieved_objects.get("triplets")
        memory_text = ""
        if triplets:
            memory_text = await super().get_context_from_objects(
                query=query,
                query_batch=query_batch,
                retrieved_objects=triplets,
            )
        if isinstance(memory_text, list):
            memory_text = "\n\n".join(memory_text)

        skills_catalog = _format_skill_catalog(retrieved_objects.get("skills") or [])
        tools_manifest = _format_tool_manifest(retrieved_objects.get("tools") or [])

        context = (
            f"# Memory\n{memory_text or '(empty)'}\n\n"
            f"# Available skills\n{skills_catalog}\n\n"
            f"# Available tools\n{tools_manifest}"
        )
        self._cached_context = context
        return context

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects=None,
        context: Any = None,
    ) -> List[Any]:
        if not isinstance(retrieved_objects, dict):
            return await super().get_completion_from_context(
                query=query,
                query_batch=query_batch,
                retrieved_objects=retrieved_objects,
                context=context,
            )

        skills: List[Skill] = retrieved_objects.get("skills") or []
        tools: List[Tool] = retrieved_objects.get("tools") or []
        tool_names = [t.name for t in tools]

        started_at_ms = int(time.time() * 1000)
        opened_skills: set[str] = set()
        opened_token = opened_skills_var.set(opened_skills)
        tool_trace: List[SkillRunToolCall] = []
        token = active_skills_var.set({s.name: s for s in skills})
        try:
            try:
                final = await self._run_tool_loop(
                    query=query,
                    initial_context=context if isinstance(context, str) else "",
                    tool_names=tool_names,
                    tool_trace=tool_trace,
                )
            except Exception as exc:
                latency_ms = int(time.time() * 1000) - started_at_ms
                skills_to_record = [s for s in skills if s.name in opened_skills]
                if skills_to_record:
                    await self._record_skill_runs(
                        skills_to_record,
                        query or "",
                        "",
                        started_at_ms=started_at_ms,
                        latency_ms=latency_ms,
                        success_score=0.0,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        tool_trace=tool_trace,
                        user=self.user,
                        dataset=self.dataset,
                        session_id=self.session_id,
                    )
                raise
        finally:
            active_skills_var.reset(token)
            opened_skills_var.reset(opened_token)

        latency_ms = int(time.time() * 1000) - started_at_ms

        # Record a SkillRun only for skills the agent actually opened via
        # load_skill — not for every skill in the prefilter catalog.
        # Ungraded agentic executions use the neutral default score until
        # explicit feedback or an evaluator supplies one.
        skills_to_record = [s for s in skills if s.name in opened_skills]
        if skills_to_record:
            await self._record_skill_runs(
                skills_to_record,
                query or "",
                final,
                started_at_ms=started_at_ms,
                latency_ms=latency_ms,
                tool_trace=tool_trace,
                user=self.user,
                dataset=self.dataset,
                session_id=self.session_id,
            )

        await self._store_session_qa(
            query=query or "",
            context=context if isinstance(context, str) else "",
            answer=final,
            triplets=retrieved_objects.get("triplets"),
        )

        return [final]

    @staticmethod
    async def _record_skill_runs(
        skills: Sequence[Skill],
        task_text: str,
        result: str,
        *,
        started_at_ms: int = 0,
        latency_ms: int = 0,
        success_score: Optional[float] = None,
        error_type: str = "",
        error_message: str = "",
        tool_trace: Optional[List[SkillRunToolCall]] = None,
        user=None,
        dataset=None,
        session_id: Optional[str] = None,
    ) -> None:
        """Persist one SkillRun node per active skill after a retrieval call."""
        from cognee.modules.engine.models import NodeSet
        from cognee.modules.engine.models.SkillRun import (
            CandidateSkill,
            SkillRun,
            UNSCORED_SKILL_RUN_SCORE,
        )
        from cognee.modules.engine.utils.generate_node_id import generate_node_id
        from cognee.modules.pipelines.models import PipelineContext
        from cognee.tasks.storage.add_data_points import add_data_points

        resolved_score = UNSCORED_SKILL_RUN_SCORE if success_score is None else success_score
        resolved_trace = list(tool_trace) if tool_trace else []
        dataset_id = getattr(dataset, "id", None)
        dataset_scope = [str(dataset_id)] if dataset_id is not None else []
        runs = [
            SkillRun(
                run_id=f"agentic:{s.id}:{uuid4()}",
                selected_skill_id=str(s.id),
                selected_skill_name=s.name,
                selected_skill=s,
                candidate_skills=[
                    CandidateSkill(
                        skill_id=str(s.id),
                        skill_name=s.name,
                        skill_description=s.description,
                        skill_text=s.skill_text or s.search_text,
                    )
                ],
                dataset_scope=dataset_scope,
                task_text=task_text,
                success_score=resolved_score,
                result_summary=(result[:500] if isinstance(result, str) else ""),
                session_id=session_id or "agentic",
                error_type=error_type,
                error_message=error_message,
                started_at_ms=started_at_ms,
                latency_ms=latency_ms,
                tool_trace=list(resolved_trace),
                belongs_to_set=[NodeSet(id=generate_node_id("NodeSet:skills"), name="skills")],
            )
            for s in skills
        ]
        ctx = None
        if user is not None and dataset is not None and dataset_id is not None:
            ctx = PipelineContext(
                user=user,
                dataset=dataset,
                data_item=SimpleNamespace(
                    id=uuid5(NAMESPACE_URL, f"cognee:skill-runs:{dataset_id}:{uuid4()}")
                ),
                pipeline_name="agentic_skill_runs_pipeline",
            )
        try:
            await add_data_points(runs, ctx=ctx)
        except Exception as exc:
            logger.warning("Failed to record SkillRun(s) after agentic retrieval: %s", exc)

    async def _get_session_history(self) -> str:
        """Return formatted session history for the active user, if caching is available."""
        if not self._use_session_cache():
            return ""

        from cognee.infrastructure.session.get_session_manager import get_session_manager

        user_id = getattr(self.user, "id", None)
        if user_id is None:
            return ""

        try:
            session_manager = get_session_manager()
            if not session_manager.is_available:
                return ""

            history = await session_manager.get_session(
                user_id=str(user_id),
                session_id=self.session_id,
                formatted=True,
                include_context=False,
            )
            return history if isinstance(history, str) else ""
        except Exception as exc:
            logger.warning("Failed to load agentic session history: %s", exc)
            return ""

    async def _store_session_qa(
        self,
        query: str,
        context: str,
        answer: str,
        triplets: Any,
    ) -> None:
        """Persist the agentic answer to the session cache when session memory is enabled."""
        if not self._use_session_cache():
            return

        from cognee.infrastructure.session.get_session_manager import get_session_manager

        user_id = getattr(self.user, "id", None)
        if user_id is None:
            return

        try:
            session_manager = get_session_manager()
            if not session_manager.is_available:
                return

            used_graph_element_ids = self._extract_context_object_ids(triplets)
            await session_manager.add_qa(
                user_id=str(user_id),
                question=query,
                context=context,
                answer=answer,
                session_id=self.session_id,
                used_graph_element_ids=used_graph_element_ids,
            )
        except Exception as exc:
            logger.warning("Failed to store agentic session QA: %s", exc)

    async def _run_tool_loop(
        self,
        query: Optional[str],
        initial_context: str,
        tool_names: List[str],
        tool_trace: Optional[List[SkillRunToolCall]] = None,
    ) -> str:
        loop_context = initial_context
        conversation_history = await self._get_session_history()

        for iteration in range(self.max_iter):
            step: AgentStep = await generate_completion(
                query=query or "",
                context=loop_context,
                user_prompt_path=self.agentic_user_prompt_path,
                system_prompt_path=self.agentic_system_prompt_path,
                conversation_history=conversation_history,
                response_model=AgentStep,
            )

            if step.final_answer is not None and step.final_answer.strip():
                return step.final_answer

            if step.tool_call is None:
                logger.warning(
                    "Agent emitted no tool_call and no final_answer on iteration %s; stopping loop",
                    iteration,
                )
                break

            t0 = time.perf_counter()
            tool_result = await self._run_tool_safely(step.tool_call, tool_names)
            if len(tool_result) > MAX_TOOL_OUTPUT_CHARS:
                tool_result = tool_result[:MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
            duration_ms = int((time.perf_counter() - t0) * 1000)
            if tool_trace is not None:
                tool_trace.append(
                    SkillRunToolCall(
                        tool_name=step.tool_call.tool_name,
                        tool_input=step.tool_call.arguments,
                        tool_output=tool_result[:1000],
                        success=not tool_result.startswith("ERROR:"),
                        duration_ms=duration_ms,
                    )
                )
            loop_context += (
                f"\n\n# Tool call {iteration + 1}: {step.tool_call.tool_name}\n"
                f"Args: {step.tool_call.arguments}\nResult:\n{tool_result}"
            )

        # Budget exhausted — force a best-effort final answer from the accumulated context.
        forced: str = await generate_completion(
            query=query or "",
            context=loop_context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            conversation_history=conversation_history,
            response_model=str,
        )
        return forced

    async def _run_tool_safely(self, call: ToolCall, tool_names: List[str]) -> str:
        try:
            result = await execute_tool(
                user=self.user,
                dataset_id=self.dataset_id,
                tool_name=call.tool_name,
                args=call.arguments,
                allowed_tools=tool_names,
            )
            return str(result) if not isinstance(result, str) else result
        except (ToolScopeError, ToolPermissionError, ToolInvocationError) as exc:
            return f"ERROR: {exc}"


def _format_skill_catalog(skills: List[Skill]) -> str:
    if not skills:
        return "(no skills loaded for this turn)"
    return "\n".join(f"- `{s.name}`: {s.description}" for s in skills)


def _format_tool_manifest(tools: List[Tool]) -> str:
    if not tools:
        return "(no tools available for this turn)"
    lines: List[str] = []
    for tool in tools:
        schema_props = tool.input_schema.get("properties", {}) if tool.input_schema else {}
        arg_list = ", ".join(schema_props.keys()) if schema_props else "no args"
        lines.append(
            f"- `{tool.name}` (perm: {tool.permission_required}): "
            f"{tool.description} [args: {arg_list}]"
        )
    return "\n".join(lines)
