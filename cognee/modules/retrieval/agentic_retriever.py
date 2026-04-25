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

from typing import Any, List, Optional, Sequence, Union
from uuid import UUID

from pydantic import BaseModel, Field

from cognee.modules.engine.models import Skill, Tool
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.tools.context import active_skills_var
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
        dataset_id: Optional[UUID] = None,
        max_iter: int = 6,
        skills_auto_retrieve: bool = False,
        skills_top_k: int = 3,
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
        self.dataset_id = dataset_id
        self.max_iter = max_iter
        self.skills_auto_retrieve = skills_auto_retrieve
        self.skills_top_k = skills_top_k
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
        triplets = await super().get_retrieved_objects(query=query, query_batch=query_batch)

        auto_query = query if self.skills_auto_retrieve else None
        skills = await resolve_skills(
            self.explicit_skills,
            dataset_id=self.dataset_id,
            auto_retrieve_query=auto_query,
            top_k=self.skills_top_k,
        )
        tools = await self._resolve_active_tools(skills)

        return {"triplets": triplets, "skills": skills, "tools": tools}

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

        memory_text = await super().get_context_from_objects(
            query=query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects.get("triplets"),
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

        token = active_skills_var.set({s.name: s for s in skills})
        try:
            final = await self._run_tool_loop(
                query=query,
                initial_context=context if isinstance(context, str) else "",
                tool_names=tool_names,
            )
        finally:
            active_skills_var.reset(token)

        # Record a SkillRun for each skill that was routed to on this
        # call. ``improve_failing_skills`` consumes only low-scored
        # runs, so ungraded agentic executions use the neutral default
        # score until explicit feedback or an evaluator supplies one.
        if skills:
            await self._record_skill_runs(skills, query or "", final)

        await self._store_session_qa(
            query=query or "",
            context=context if isinstance(context, str) else "",
            answer=final,
            triplets=retrieved_objects.get("triplets"),
        )

        return [final]

    @staticmethod
    async def _record_skill_runs(skills: Sequence[Skill], task_text: str, result: str) -> None:
        """Persist one SkillRun node per active skill after a retrieval call."""
        from cognee.modules.engine.models.SkillRun import SkillRun, UNSCORED_SKILL_RUN_SCORE
        from cognee.tasks.storage.add_data_points import add_data_points

        runs = [
            SkillRun(
                run_id=f"agentic:{s.name}:{id(result)}",
                selected_skill_id=s.name,
                task_text=task_text,
                success_score=UNSCORED_SKILL_RUN_SCORE,
                result_summary=(result[:500] if isinstance(result, str) else ""),
                session_id="agentic",
            )
            for s in skills
        ]
        try:
            await add_data_points(runs)
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

            tool_result = await self._run_tool_safely(step.tool_call, tool_names)
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
