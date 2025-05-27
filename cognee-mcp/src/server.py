import json
import os
import sys
import argparse
import cognee
import asyncio
import time
from datetime import datetime
from cognee.shared.logging_utils import get_logger, get_log_file_location
import importlib.util
from contextlib import redirect_stdout
import mcp.types as types
from mcp.server import FastMCP
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
from cognee.modules.users.methods import get_default_user
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.storage.utils import JSONEncoder

mcp = FastMCP("Cognee")

logger = get_logger()
log_file = get_log_file_location()

@mcp.tool()
async def getUserContext(user_query: str) -> list[types.TextContent]:
    """
    Automatically retrieves relevant user context for the current query.
    Called at conversation start - no user prompt needed.
    """
    with redirect_stdout(sys.stderr):
        logger.info(f"Executing getUserContext for user_query: '{user_query}'")

        user_id = os.environ.get("USER_ID", "tyrone")
        
        # Single comprehensive search to get user profile
        try:
            results = await cognee.search(
                query_type=SearchType.INSIGHTS,
                query_text=f"{user_id} profile preferences interests goals communication style",
                top_k=10
            )
            
            if results:
                profile_summary = retrieved_edges_to_string(results)
                logger.info(f"User profile found for {user_id}")
                
                llm_instruction_prompt = (
                    f"User Query: '{user_query}'\n\n"
                    f"User Profile ({user_id}):\n"
                    f"{profile_summary}\n\n"
                    f"Instructions: Answer the user's query while considering their profile. "
                    f"Tailor your response to their preferences, interests, and communication style."
                )
            else:
                logger.info(f"No profile information found for user {user_id}")
                llm_instruction_prompt = (
                    f"User Query: '{user_query}'\n\n"
                    f"Note: No specific profile information found for user {user_id}. "
                    f"Please provide a helpful, general response."
                )
                
        except Exception as e:
            logger.error(f"Error fetching profile: {str(e)}")
            llm_instruction_prompt = (
                f"User Query: '{user_query}'\n\n"
                f"Error retrieving profile: {str(e)}\n"
                f"Please provide a helpful general response."
            )
            
        logger.info(f"Constructed LLM prompt. Length: {len(llm_instruction_prompt)} chars")
        return [types.TextContent(type="text", text=llm_instruction_prompt)]


@mcp.tool()
async def saveMemory(content: str, memory_type: str, importance: str = "medium") -> list[types.TextContent]:
    """
    Saves user information ONLY after Claude prompts and user confirms.
    Claude should ask: "This seems important - should I remember that you [preference/goal]?"
    
    Args:
        content: The information to save
        memory_type: One of ["preference", "goal", "communication_style", "expertise", "project", "general"]
        importance: One of ["low", "medium", "high", "critical"]
    """
    async def save_memory_task(content: str, memory_type: str, importance: str):
        with redirect_stdout(sys.stderr):
            logger.info(f"Memory save process starting: {memory_type} ({importance})")
            
            user_id = os.environ.get("USER_ID", "tyrone")
            
            try:
                # Format the memory with metadata using proper timestamp
                current_time = datetime.now().isoformat()
                formatted_memory = f"""
USER: {user_id}
TYPE: {memory_type}
IMPORTANCE: {importance}
TIMESTAMP: {current_time}
CONTENT: {content}
"""
                
                # Add to Cognee with memory tag
                await cognee.add(formatted_memory, node_set=["memory"])
                
                # Use ontology for proper graph structure
                ontology_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "..",
                    "examples",
                    "python", 
                    "ontology_input_example",
                    "profile_ontology.owl"
                )
                
                if not os.path.exists(ontology_path):
                    ontology_path = "/Users/tyroneavnit/Projects/cognee/examples/python/ontology_input_example/profile_ontology.owl"
                
                # Full cognify processing to integrate into knowledge graph
                await cognee.cognify(ontology_file_path=ontology_path)
                
                logger.info(f"Memory saved successfully: {memory_type} ({importance})")
                
            except Exception as e:
                logger.error(f"Memory save failed: {str(e)}")
                raise ValueError(f"Failed to save memory: {str(e)}")

    with redirect_stdout(sys.stderr):
        logger.info(f"Launching memory save: {memory_type} ({importance}) - {content[:100]}...")
        
    # Launch background task
    asyncio.create_task(save_memory_task(content, memory_type, importance))

    response = (
        f"Background memory save process launched due to MCP timeout limitations.\n"
        f"Memory Details:\n"
        f"- Type: {memory_type}\n"
        f"- Importance: {importance}\n"
        f"- Content: {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
        f"To check current save status use the cognify_status tool\n"
        f"or check the log file at: {log_file}\n\n"
        f"This memory will be available for future conversations once processing completes."
    )
    
    return [types.TextContent(type="text", text=response)]


@mcp.tool()
async def updateMemory(memory_id: str, new_content: str) -> list[types.TextContent]:
    """
    Updates existing memory entry after user confirmation.
    
    Args:
        memory_id: ID of the memory to update
        new_content: Updated information
    """
    with redirect_stdout(sys.stderr):
        logger.info(f"Updating memory {memory_id}")
        
        # Implementation depends on how Cognee handles updates
        # For now, simple approach: save new version with update flag
        response = f"Memory update functionality - requires Cognee update API implementation"
        return [types.TextContent(type="text", text=response)]


@mcp.tool()
async def removeMemory(memory_id: str) -> list[types.TextContent]:
    """
    Removes a memory entry after user confirmation.
    
    Args:
        memory_id: ID of the memory to remove
    """
    with redirect_stdout(sys.stderr):
        logger.info(f"Removing memory {memory_id}")
        
        # Implementation depends on how Cognee handles deletions
        response = f"Memory removal functionality - requires Cognee delete API implementation"
        return [types.TextContent(type="text", text=response)]


@mcp.tool()
async def retrieveMemories(query: str = "", memory_type: str = "", importance: str = "") -> list[types.TextContent]:
    """
    Retrieves saved memories based on search criteria.
    
    Args:
        query: Search query to find relevant memories (optional)
        memory_type: Filter by memory type (optional)
        importance: Filter by importance level (optional)
    """
    with redirect_stdout(sys.stderr):
        logger.info(f"Retrieving memories: query='{query}', type='{memory_type}', importance='{importance}'")
        
        user_id = os.environ.get("USER_ID", "tyrone")
        
        # Build search query
        search_parts = [f"USER {user_id}"]
        if memory_type:
            search_parts.append(f"TYPE {memory_type}")
        if importance:
            search_parts.append(f"IMPORTANCE {importance}")
        if query:
            search_parts.append(query)
            
        search_query = " ".join(search_parts)
        
        try:
            search_results = await cognee.search(
                query_type=SearchType.INSIGHTS,
                query_text=search_query,
                datasets=["memory"],  # Search specifically in memory-tagged data
                top_k=20
            )
            
            if search_results:
                response = f"Found memories for {user_id}:\n\n" + retrieved_edges_to_string(search_results)
            else:
                response = f"No memories found matching the criteria."
                
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            response = f"Error retrieving memories: {str(e)}"
        
        return [types.TextContent(type="text", text=response)]


@mcp.tool()
async def search(search_query: str, search_type: str) -> list[types.TextContent]:
    """
    Search the knowledge graph using various search strategies.
    
    Args:
        search_query: The search query text
        search_type: One of the available SearchType enum values
        
    Available SearchTypes:
        - SUMMARIES: Get high-level summaries of information
        - INSIGHTS: Extract meaningful insights and relationships
        - CHUNKS: Retrieve raw text chunks
        - RAG_COMPLETION: RAG-based completion responses
        - GRAPH_COMPLETION: Graph-enhanced completions
        - GRAPH_SUMMARY_COMPLETION: Completions with graph summaries
        - CODE: Search code-specific information
        - CYPHER: Execute Cypher queries on graph
        - NATURAL_LANGUAGE: Natural language search interface
    """
    async def search_task(search_query: str, search_type: str) -> str:
        with redirect_stdout(sys.stderr):
            search_results = await cognee.search(
                query_type=SearchType[search_type.upper()], query_text=search_query
            )

            if search_type.upper() == "CODE":
                return json.dumps(search_results, cls=JSONEncoder)
            elif search_type.upper() in ["GRAPH_COMPLETION", "RAG_COMPLETION"]:
                return search_results[0] if search_results else ""
            elif search_type.upper() == "CHUNKS":
                return str(search_results)
            elif search_type.upper() == "INSIGHTS":
                return retrieved_edges_to_string(search_results)
            else:
                return str(search_results)

    search_results = await search_task(search_query, search_type)
    return [types.TextContent(type="text", text=search_results)]


@mcp.tool()
async def cognify(text: str, graph_model_file: str = None, graph_model_name: str = None) -> list[types.TextContent]:
    """Build knowledge graph from the input text"""
    async def cognify_task(
        text: str, graph_model_file: str = None, graph_model_name: str = None
    ) -> str:
        with redirect_stdout(sys.stderr):
            logger.info("Cognify process starting.")
            if graph_model_file and graph_model_name:
                graph_model = load_class(graph_model_file, graph_model_name)
            else:
                graph_model = KnowledgeGraph

            await cognee.add(text)

            try:
                await cognee.cognify(graph_model=graph_model)
                logger.info("Cognify process finished.")
            except Exception as e:
                logger.error("Cognify process failed.")
                raise ValueError(f"Failed to cognify: {str(e)}")

    asyncio.create_task(
        cognify_task(
            text=text,
            graph_model_file=graph_model_file,
            graph_model_name=graph_model_name,
        )
    )

    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current cognify status use the cognify_status tool\n"
        f"or check the log file at: {log_file}"
    )

    return [types.TextContent(type="text", text=text)]


@mcp.tool()
async def codify(repo_path: str) -> list[types.TextContent]:
    """Build code graph from repository"""
    async def codify_task(repo_path: str):
        with redirect_stdout(sys.stderr):
            logger.info("Codify process starting.")
            results = []
            async for result in run_code_graph_pipeline(repo_path, False):
                results.append(result)
                logger.info(result)
            if all(results):
                logger.info("Codify process finished successfully.")
            else:
                logger.info("Codify process failed.")

    asyncio.create_task(codify_task(repo_path))

    text = (
        f"Background process launched due to MCP timeout limitations.\n"
        f"To check current codify status use the codify_status tool\n"
        f"or you can check the log file at: {log_file}"
    )

    return [types.TextContent(type="text", text=text)]


@mcp.tool()
async def prune() -> list[types.TextContent]:
    """Reset the knowledge graph"""
    with redirect_stdout(sys.stderr):
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        return [types.TextContent(type="text", text="Pruned")]


@mcp.tool()
async def cognify_status() -> list[types.TextContent]:
    """Get status of cognify pipeline"""
    with redirect_stdout(sys.stderr):
        user = await get_default_user()
        status = await get_pipeline_status(
            [await get_unique_dataset_id("main_dataset", user)], "cognify_pipeline"
        )
        return [types.TextContent(type="text", text=str(status))]


@mcp.tool()
async def codify_status() -> list[types.TextContent]:
    """Get status of codify pipeline"""
    with redirect_stdout(sys.stderr):
        user = await get_default_user()
        status = await get_pipeline_status(
            [await get_unique_dataset_id("codebase", user)], "cognify_code_pipeline"
        )
        return [types.TextContent(type="text", text=str(status))]


def node_to_string(node):
    """Convert a node to a string representation"""
    node_data = ", ".join(
        [f'{key}: "{value}"' for key, value in node.items() if key in ["id", "name"]]
    )
    return f"Node({node_data})"


def retrieved_edges_to_string(search_results):
    """Convert search results (triplets) to a human-readable profile summary"""
    if not search_results:
        return "No profile information found."
    
    # Parse the graph data to extract meaningful information
    profile_info = {
        'preferences': [],
        'interests': [],
        'goals': [],
        'values': [],
        'communication_style': [],
        'learning_style': [],
        'expertise': [],
        'background': []
    }
    
    for triplet in search_results:
        node1, edge, node2 = triplet
        relationship = edge.get("relationship_name", "").lower()
        
        # Extract user name
        user_name = None
        if node1.get("name") and "user" in str(node1.get("name", "")).lower():
            user_name = node1.get("name")
        elif node2.get("name") and "user" in str(node2.get("name", "")).lower():
            user_name = node2.get("name")
            
        # Parse different types of relationships
        if "favorite" in relationship or "prefers" in relationship:
            if node2.get("name"):
                profile_info['preferences'].append(f"Favorite {relationship.replace('favorite_', '').replace('_', ' ')}: {node2['name']}")
        
        elif "interested" in relationship or "likes" in relationship:
            if node2.get("name"):
                profile_info['interests'].append(node2['name'])
                
        elif "goal" in relationship or "wants" in relationship or "aims" in relationship:
            if node2.get("name"):
                profile_info['goals'].append(node2['name'])
                
        elif "values" in relationship or "believes" in relationship:
            if node2.get("name"):
                profile_info['values'].append(node2['name'])
                
        elif "communication" in relationship or "style" in relationship:
            if node2.get("name"):
                profile_info['communication_style'].append(node2['name'])
                
        elif "learning" in relationship or "learns" in relationship:
            if node2.get("name"):
                profile_info['learning_style'].append(node2['name'])
                
        elif "expert" in relationship or "skilled" in relationship or "knows" in relationship:
            if node2.get("name"):
                profile_info['expertise'].append(node2['name'])
                
        elif "background" in relationship or "experience" in relationship:
            if node2.get("name"):
                profile_info['background'].append(node2['name'])
    
    # Build human-readable summary
    summary_parts = []
    
    if profile_info['preferences']:
        summary_parts.append(f"Preferences: {', '.join(profile_info['preferences'])}")
        
    if profile_info['interests']:
        summary_parts.append(f"Interests: {', '.join(profile_info['interests'])}")
        
    if profile_info['goals']:
        summary_parts.append(f"Goals: {', '.join(profile_info['goals'])}")
        
    if profile_info['values']:
        summary_parts.append(f"Values: {', '.join(profile_info['values'])}")
        
    if profile_info['communication_style']:
        summary_parts.append(f"Communication Style: {', '.join(profile_info['communication_style'])}")
        
    if profile_info['learning_style']:
        summary_parts.append(f"Learning Style: {', '.join(profile_info['learning_style'])}")
        
    if profile_info['expertise']:
        summary_parts.append(f"Expertise: {', '.join(profile_info['expertise'])}")
        
    if profile_info['background']:
        summary_parts.append(f"Background: {', '.join(profile_info['background'])}")
    
    if summary_parts:
        return "\n".join(summary_parts)
    else:
        # Fallback: if we can't parse meaningful info, show a simplified version
        simple_facts = []
        for triplet in search_results:
            node1, edge, node2 = triplet
            relationship = edge.get("relationship_name", "")
            if node1.get("name") and node2.get("name"):
                simple_facts.append(f"{node1['name']} {relationship} {node2['name']}")
        
        return "\n".join(simple_facts[:5])  # Limit to 5 most relevant facts


def load_class(model_file, model_name):
    """Dynamically load a class from a file"""
    model_file = os.path.abspath(model_file)
    spec = importlib.util.spec_from_file_location("graph_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, model_name)
    return model_class


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport to use for communication with the client. (default: stdio)",
    )

    args = parser.parse_args()

    logger.info(f"Starting MCP server with transport: {args.transport}")
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info(
            f"Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}"
        )
        await mcp.run_sse_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error initializing Cognee MCP server: {str(e)}")
        raise
