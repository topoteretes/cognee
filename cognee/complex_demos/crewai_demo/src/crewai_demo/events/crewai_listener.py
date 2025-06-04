import time
from uuid import uuid4
from crewai.utilities.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
)
from crewai.utilities.events.base_event_listener import BaseEventListener

from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunActivity
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue


class CrewAIListener(BaseEventListener):
    def __init__(self, pipeline_run_id):
        super().__init__()
        self.pipeline_run_id = pipeline_run_id

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            push_to_queue(
                self.pipeline_run_id,
                PipelineRunActivity(
                    pipeline_run_id=self.pipeline_run_id,
                    payload={
                        "id": str(uuid4()),
                        "timestamp": time.time() * 1000,
                        "activity": f"Crew '{event.crew_name}' has started execution",
                    },
                ),
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            push_to_queue(
                self.pipeline_run_id,
                PipelineRunActivity(
                    pipeline_run_id=self.pipeline_run_id,
                    payload={
                        "id": str(uuid4()),
                        "timestamp": time.time() * 1000,
                        "activity": f"Crew '{event.crew_name}' has completed execution",
                    },
                ),
            )

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionStartedEvent):
            push_to_queue(
                self.pipeline_run_id,
                PipelineRunActivity(
                    pipeline_run_id=self.pipeline_run_id,
                    payload={
                        "id": str(uuid4()),
                        "timestamp": time.time() * 1000,
                        "activity": f"Agent '{event.agent.role}' started execution",
                    },
                ),
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            push_to_queue(
                self.pipeline_run_id,
                PipelineRunActivity(
                    pipeline_run_id=self.pipeline_run_id,
                    payload={
                        "id": str(uuid4()),
                        "timestamp": time.time() * 1000,
                        "activity": f"Agent '{event.agent.role}' completed execution",
                    },
                ),
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_agent_execution_completed(source, event: ToolUsageStartedEvent):
            push_to_queue(
                self.pipeline_run_id,
                PipelineRunActivity(
                    pipeline_run_id=self.pipeline_run_id,
                    payload={
                        "id": str(uuid4()),
                        "timestamp": time.time() * 1000,
                        "activity": f"Agent tool call ({event.tool_name}) execution started",
                    },
                ),
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_agent_execution_completed(source, event: ToolUsageFinishedEvent):
            push_to_queue(
                self.pipeline_run_id,
                PipelineRunActivity(
                    pipeline_run_id=self.pipeline_run_id,
                    payload={
                        "id": str(uuid4()),
                        "timestamp": time.time() * 1000,
                        "activity": f"Agent tool call ({event.tool_name}) execution completed",
                    },
                ),
            )
