import os
import logging
import cognee
import asyncio

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from dotenv import load_dotenv
from cognee.api.v1.search import SearchType
from cognee.modules.engine.models import NodeSet
from cognee.shared.logging_utils import setup_logging


load_dotenv()

os.environ["LLM_API_KEY"] = ""
# Notes: Nodesets cognee feature only works with kuzu and Neo4j graph databases
os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"


class ProcurementMemorySystem:
    """Procurement system with persistent memory using Cognee"""

    async def setup_memory_data(self):
        """Load and store procurement data in memory"""

        # Procurement system dummy data
        vendor_conversation_text_techsupply = """
        Assistant: Hello! This is Sarah from TechSupply Solutions.
        Thanks for reaching out for your IT procurement needs.

        User: We're looking to procure 50 high-performance enterprise laptops.
        Specs: Intel i7, 16GB RAM, 512GB SSD, dedicated graphics card.
        Budget: $80,000. What models do you have?

        Assistant: TechSupply Solutions can offer Dell Precision 5570 ($1,450) and Lenovo ThinkPad P1 ($1,550).
        Both come with a 3-year warranty. Delivery: 2–3 weeks (Dell), 3–4 weeks (Lenovo).

        User: Do you provide bulk discounts? We're planning another 200 units next quarter.

        Assistant: Yes! Orders over $50,000 get 8% off.
        So for your current order:
        - Dell = $1,334 each ($66,700 total)
        - Lenovo = $1,426 each ($71,300 total)

        And for 200 units next quarter, we can offer 12% off with flexible delivery.
        """

        vendor_conversation_text_office_solutions = """
        Assistant: Hi, this is Martin from vendor Office Solutions. How can we assist you?

        User: We need 50 laptops for our engineers.
        Specs: i7 CPU, 16GB RAM, 512GB SSD, dedicated GPU.
        We can spend up to $80,000. Can you meet this?

        Assistant: Office Solutions can offer HP ZBook Power G9 for $1,600 each.
        Comes with 2-year warranty, delivery time is 4–5 weeks.

        User: That's a bit long — any options to speed it up?

        Assistant: We can expedite for $75 per unit, bringing delivery to 3–4 weeks.
        Also, for orders over $60,000 we give 6% off.

        So:
        - Base price = $1,600 → $1,504 with discount
        - Expedited price = $1,579

        User: Understood. Any room for better warranty terms?

        Assistant: We’re working on adding a 3-year warranty option next quarter for enterprise clients.
        """

        previous_purchases_text = """
        Previous Purchase Records:
        1. Vendor: TechSupply Solutions
           Item: Desktop computers - 25 units
           Amount: $35,000
           Date: 2024-01-15
           Performance: Excellent delivery, good quality, delivered 2 days early
           Rating: 5/5
           Notes: Responsive support team, competitive pricing

        2. Vendor: Office Solutions
           Item: Office furniture
           Amount: $12,000
           Date: 2024-02-20
           Performance: Delayed delivery by 1 week, average quality
           Rating: 2/5
           Notes: Poor communication, but acceptable product quality
        """

        procurement_preferences_text = """
        Procurement Policies and Preferences:
        1. Preferred vendors must have 3+ year warranty coverage
        2. Maximum delivery time: 30 days for non-critical items
        3. Bulk discount requirements: minimum 5% for orders over $50,000
        4. Prioritize vendors with sustainable/green practices
        5. Vendor rating system: require minimum 4/5 rating for new contracts
        """

        # Initializing and pruning databases
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        # Store data in different memory categories
        await cognee.add(
            data=[vendor_conversation_text_techsupply, vendor_conversation_text_office_solutions],
            node_set=["vendor_conversations"],
        )

        await cognee.add(data=previous_purchases_text, node_set=["purchase_history"])

        await cognee.add(data=procurement_preferences_text, node_set=["procurement_policies"])

        # Process all data through Cognee's knowledge graph
        await cognee.cognify()

    async def search_memory(self, query, search_categories=None):
        """Search across different memory layers"""
        results = {}
        for category in search_categories:
            category_results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                node_type=NodeSet,
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
            print(f"Answer: \n{top_answer.strip()}\n")
            research_notes[category].append({"question": q, "answer": top_answer})

    print("Contextual research complete.\n")

    print("Compiling structured research information for decision-making...\n")
    research_information = "\n\n".join(
        f"Q: {note['question']}\nA: {note['answer'].strip()}"
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
