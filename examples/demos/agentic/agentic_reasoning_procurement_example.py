# ruff: noqa: E402
import asyncio
from pathlib import Path
import logging
import os

from dotenv import load_dotenv

load_dotenv()

# Notes: Nodesets cognee feature only works with Ladybug and Neo4j graph databases
# Set os.environ before importing Cognee: Cognee reads env-backed settings at import time, so values
# assigned later may not override defaults or `.env`. See https://docs.cognee.ai/setup-configuration/overview#using-os-environ
os.environ["GRAPH_DATABASE_PROVIDER"] = "ladybug"

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402
from cognee.infrastructure.llm.LLMGateway import LLMGateway  # noqa: E402
from cognee.shared.logging_utils import setup_logging  # noqa: E402


class ProcurementMemorySystem:
    """Procurement system with persistent memory using Cognee"""

    async def setup_memory_data(self):
        """Load and store procurement data in memory"""

        # Procurement system dummy data
        data_dir = Path(__file__).parent / "agentic_reasoning_procurement_example_data"
        vendor_conversation_text_techsupply = (data_dir / "techsupply_conversation.txt").read_text()

        vendor_conversation_text_office_solutions = (
            data_dir / "office_solutions_conversation.txt"
        ).read_text()

        previous_purchases_text = (data_dir / "purchase_history.txt").read_text()

        procurement_preferences_text = (data_dir / "procurement_policies.txt").read_text()

        # Initializing and pruning databases
        await cognee.forget(everything=True)

        # Store data in different memory categories
        await cognee.remember(
            data=[vendor_conversation_text_techsupply, vendor_conversation_text_office_solutions],
            node_set=["vendor_conversations"],
            self_improvement=False,
        )

        await cognee.remember(
            data=previous_purchases_text,
            node_set=["purchase_history"],
            self_improvement=False,
        )

        await cognee.remember(
            data=procurement_preferences_text,
            node_set=["procurement_policies"],
            self_improvement=False,
        )

    async def search_memory(self, query, search_categories=None):
        """Search across different memory layers"""
        results = {}
        for category in search_categories:
            category_results = await cognee.recall(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                node_name=[category],
                top_k=30,
            )
            results[category] = category_results

        return results


async def run_procurement_example():
    """Main function demonstrating procurement memory system"""
    print("Building AI Procurement System with Memory: Cognee Integration...\n")

    # Initialize the procurement memory system
    procurement_system = ProcurementMemorySystem()

    # Setup memory with procurement data
    print("Setting up procurement memory data...")
    await procurement_system.setup_memory_data()
    print("Memory successfully populated and processed.\n")

    research_questions = {
        "vendor_conversations": [
            "What are the laptops that are discussed, together with their vendors?",
            "What pricing was offered by each vendor before and after discounts?",
            "What were the delivery time estimates for each product?",
        ],
        "purchase_history": [
            "Which vendors have we worked with in the past?",
            "What were the satisfaction ratings for each vendor?",
            "Were there any complaints or red flags associated with specific vendors?",
        ],
        "procurement_policies": [
            "What are our company’s bulk discount requirements?",
            "What is the maximum acceptable delivery time for non-critical items?",
            "What is the minimum vendor rating for new contracts?",
        ],
    }

    research_notes = {}
    print("Running contextual research questions...\n")
    for category, questions in research_questions.items():
        print(f"Category: {category}")
        research_notes[category] = []
        for q in questions:
            print(f"Question: \n{q}")
            results = await procurement_system.search_memory(q, search_categories=[category])
            top_answer = results[category][0]
            print(f"Answer: \n{top_answer}\n")
            research_notes[category].append({"question": q, "answer": top_answer})

    print("Contextual research complete.\n")

    print("Compiling structured research information for decision-making...\n")
    research_information = "\n\n".join(
        f"Q: {note['question']}\nA: {note['answer']}"
        for section in research_notes.values()
        for note in section
    )

    print("Compiled Research Summary:\n")
    print(research_information)
    print("\nPassing research to LLM for final procurement recommendation...\n")

    final_decision = await LLMGateway.acreate_structured_output(
        text_input=research_information,
        system_prompt="""You are a procurement decision assistant. Use the provided QA pairs that were collected through a research phase. Recommend the best vendor,
         based on pricing, delivery, warranty, policy fit, and past performance. Be concise and justify your choice with evidence.
         """,
        response_model=str,
    )

    print("Final Decision:")
    print(final_decision.strip())


# Run the example
if __name__ == "__main__":
    setup_logging(logging.ERROR)
    asyncio.run(run_procurement_example())
