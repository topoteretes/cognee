import asyncio
import logging
from services.memory.cognee_service import remember_data, recall_data
from services.timeline.events import log_event

logger = logging.getLogger("agents_sim")

AGENT_PROFILES = {
    "Architect": {
        "name": "Alex",
        "role": "Architect",
        "avatar": "📐",
        "color": "#3B82F6" # Blue
    },
    "Developer": {
        "name": "Devin",
        "role": "Developer",
        "avatar": "💻",
        "color": "#10B981" # Green
    },
    "QA": {
        "name": "Quinn",
        "role": "QA Engineer",
        "avatar": "🔍",
        "color": "#EF4444" # Red
    },
    "Researcher": {
        "name": "Regina",
        "role": "Researcher",
        "avatar": "🔬",
        "color": "#8B5CF6" # Purple
    },
    "TechWriter": {
        "name": "Wendy",
        "role": "Technical Writer",
        "avatar": "✍️",
        "color": "#F59E0B" # Amber
    }
}

SIMULATION_STEPS = [
    {
        "agent": "Architect",
        "action": "write",
        "topic": "Authentication Specs",
        "content": "Authentication Specification: MemoryOS requires JWT token-based auth. Tokens must use the HS256 signature algorithm. User metadata must store username, role, and workspace_id. Save passwords hashed in a database using bcrypt.",
        "log": "Alex (Architect) defined the core JWT specifications and cryptographic requirements for MemoryOS authentication."
    },
    {
        "agent": "Developer",
        "action": "read",
        "query": "JWT token HS256 user metadata specifications",
        "log": "Devin (Developer) recalled the Architect specification from shared memory to build the implementation model."
    },
    {
        "agent": "Developer",
        "action": "write",
        "topic": "FastAPI Login Code",
        "content": "FastAPI Login Code: Implemented `/api/auth/login` endpoint using PyJWT for token generation and bcrypt for password verification. The tokens carry username and role. Inactive users are not checked yet.",
        "log": "Devin (Developer) coded and registered the FastAPI login endpoint to the memory graph."
    },
    {
        "agent": "QA",
        "action": "read",
        "query": "FastAPI login endpoint implementation",
        "log": "Quinn (QA) recalled Devin's implementation details from the shared graph to run verification cases."
    },
    {
        "agent": "QA",
        "action": "write",
        "topic": "Security Vulnerability Bug",
        "content": "Security Bug: The `/api/auth/login` endpoint fails to verify if a user account is active. Inactive or suspended users can still login and receive valid JWT tokens.",
        "log": "Quinn (QA) identified a critical vulnerability and saved the bug report into the shared memory space."
    },
    {
        "agent": "Researcher",
        "action": "read",
        "query": "login endpoint security bug inactive users",
        "log": "Regina (Researcher) searched the memory graph for the active security report to search for a mitigation strategy."
    },
    {
        "agent": "Researcher",
        "action": "write",
        "topic": "Security Mitigation Guidelines",
        "content": "Security Guidelines: To prevent suspended accounts from logging in, check the database flag `user.is_active` prior to token generation. If false, immediately raise HTTP 403 Forbidden.",
        "log": "Regina (Researcher) injected industry security standards for JWT auth into the memory base."
    },
    {
        "agent": "Developer",
        "action": "read",
        "query": "security guidelines user.is_active HTTP 403",
        "log": "Devin (Developer) recalled the researcher's guidelines to design a hotfix."
    },
    {
        "agent": "Developer",
        "action": "write",
        "topic": "FastAPI Login Fix",
        "content": "Auth Hotfix: Updated `/api/auth/login` to query `is_active` from the user table. Suspended users now fail authentication with HTTP 403 Forbidden. Verified test passes.",
        "log": "Devin (Developer) committed and registered the auth bug fix to the shared graph."
    },
    {
        "agent": "TechWriter",
        "action": "read",
        "query": "Authentication Specification FastAPI Login login endpoint fixes",
        "log": "Wendy (Technical Writer) searched the entire auth timeline to compile user docs."
    },
    {
        "agent": "TechWriter",
        "action": "write",
        "topic": "API Auth Docs",
        "content": "MemoryOS API Auth Docs: Access endpoints via `/api/auth/login`. Returns JWT tokens using HS256. Safe hashing handles passwords using bcrypt. Accounts with `is_active=False` receive HTTP 403.",
        "log": "Wendy (Technical Writer) published the final API authentication documentation to the shared knowledge base."
    }
]

# In-memory simulation status
class SimulationManager:
    def __init__(self):
        self.current_step = 0
        self.history = []
        self.is_running = False

    def reset(self):
        self.current_step = 0
        self.history = []
        self.is_running = False
        log_event("AgentSimReset", "Collaboration workspace simulation reset")
        return {"status": "reset", "step": 0}

    async def execute_next_step(self) -> dict:
        """Executes the next step in the simulation loop."""
        if self.current_step >= len(SIMULATION_STEPS):
            return {"status": "completed", "message": "Simulation already completed."}
            
        self.is_running = True
        step = SIMULATION_STEPS[self.current_step]
        agent_role = step["agent"]
        profile = AGENT_PROFILES[agent_role]
        
        # Log action event
        log_event("AgentAction", f"{profile['name']} ({profile['role']}): {step['log']}", {
            "agent": profile["name"],
            "role": profile["role"],
            "action": step["action"]
        })
        
        recall_context = []
        
        try:
            if step["action"] == "write":
                # Ingest data to Cognee under agent dataset
                await remember_data(
                    data=step["content"],
                    dataset_name="agent_collab_space"
                )
            elif step["action"] == "read":
                # Query Cognee
                results = await recall_data(
                    query=step["query"],
                    query_type_str="GRAPH_COMPLETION",
                    dataset_name="agent_collab_space"
                )
                recall_context = [r.get("text", str(r)) for r in results]
                
        except Exception as e:
            logger.error(f"Error executing agent step: {e}", exc_info=True)
            # Log error but don't break mock simulation visualizer
            pass
            
        step_result = {
            "step_index": self.current_step,
            "agent": profile,
            "action": step["action"],
            "topic": step.get("topic", "Recall"),
            "content": step.get("content", f"Queried: \"{step.get('query')}\""),
            "log": step["log"],
            "recall_context": recall_context
        }
        
        self.history.append(step_result)
        self.current_step += 1
        
        if self.current_step >= len(SIMULATION_STEPS):
            self.is_running = False
            log_event("AgentSimCompleted", "Multi-Agent collaboration simulation finished")
            
        return {
            "status": "running" if self.current_step < len(SIMULATION_STEPS) else "completed",
            "step_index": self.current_step,
            "step": step_result
        }

sim_manager = SimulationManager()
