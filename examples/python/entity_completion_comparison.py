import cognee
import asyncio
import logging

from cognee.api.v1.search import SearchType
from cognee.shared.utils import setup_logging
from cognee.modules.retrieval.EntityCompletionRetriever import EntityCompletionRetriever
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)
from cognee.modules.retrieval.context_providers.SummarizedTripletSearchContextProvider import (
    SummarizedTripletSearchContextProvider,
)
from cognee.modules.retrieval.entity_extractors.DummyEntityExtractor import DummyEntityExtractor

article_1 = """
Title: The Theory of Relativity: A Revolutionary Breakthrough
Author: Dr. Sarah Chen

Albert Einstein's theory of relativity fundamentally changed our understanding of space, time, and gravity. Published in 1915, the general theory of relativity describes gravity as a consequence of the curvature of spacetime caused by mass and energy. This groundbreaking work built upon his special theory of relativity from 1905, which introduced the famous equation E=mc².

Einstein's work at the Swiss Patent Office gave him time to develop these revolutionary ideas. His mathematical framework predicted several phenomena that were later confirmed, including:
- The bending of light by gravity
- The precession of Mercury's orbit
- The existence of black holes

The theory continues to be tested and validated today, most recently through the detection of gravitational waves by LIGO in 2015, exactly 100 years after its publication.
"""

article_2 = """
Title: The Manhattan Project and Its Scientific Director
Author: Prof. Michael Werner

J. Robert Oppenheimer's leadership of the Manhattan Project marked a pivotal moment in scientific history. As scientific director of the Los Alamos Laboratory, he assembled and led an extraordinary team of physicists in the development of the atomic bomb during World War II.

Oppenheimer's journey to Los Alamos began at Harvard and continued through his groundbreaking work in quantum mechanics and nuclear physics at Berkeley. His expertise in theoretical physics and exceptional leadership abilities made him the ideal candidate to head the secret weapons laboratory.

Key aspects of his directorship included:
- Recruitment of top scientific talent from across the country
- Integration of theoretical physics with practical engineering challenges
- Development of implosion-type nuclear weapons
- Management of complex security and ethical considerations

After witnessing the first nuclear test, codenamed Trinity, Oppenheimer famously quoted the Bhagavad Gita: "Now I am become Death, the destroyer of worlds." This moment reflected the profound moral implications of scientific advancement that would shape his later advocacy for international atomic controls.
"""

article_3 = """
Title: The Birth of Quantum Physics
Author: Dr. Lisa Martinez

The early 20th century witnessed a revolutionary transformation in our understanding of the microscopic world. The development of quantum mechanics emerged from the collaborative efforts of numerous brilliant physicists grappling with phenomena that classical physics couldn't explain.

Key contributors and their insights included:
- Max Planck's discovery of energy quantization (1900)
- Niels Bohr's model of the atom with discrete energy levels (1913)
- Werner Heisenberg's uncertainty principle (1927)
- Erwin Schrödinger's wave equation (1926)
- Paul Dirac's quantum theory of the electron (1928)

Einstein's 1905 paper on the photoelectric effect, which demonstrated light's particle nature, was a crucial contribution to this field. The Copenhagen interpretation, developed primarily by Bohr and Heisenberg, became the standard understanding of quantum mechanics, despite ongoing debates about its philosophical implications. These foundational developments continue to influence modern physics, from quantum computing to quantum field theory.
"""


async def main(enable_steps):
    # Step 1: Reset data and system state
    if enable_steps.get("prune_data"):
        await cognee.prune.prune_data()
        print("Data pruned.")

    if enable_steps.get("prune_system"):
        await cognee.prune.prune_system(metadata=True)
        print("System pruned.")

    # Step 2: Add text
    if enable_steps.get("add_text"):
        text_list = [article_1, article_2, article_3]
        for text in text_list:
            await cognee.add(text)
            print(f"Added text: {text[:50]}...")

    # Step 3: Create knowledge graph
    if enable_steps.get("cognify"):
        await cognee.cognify()
        print("Knowledge graph created.")

    # Step 4: Query insights using our new retrievers
    if enable_steps.get("retriever"):
        # Common settings
        search_settings = {
            "top_k": 5,
            "collections": ["Entity_name", "TextSummary_text"],
            "properties_to_project": ["name", "description", "text"],
        }

        # Create both context providers
        direct_provider = TripletSearchContextProvider(**search_settings)
        summary_provider = SummarizedTripletSearchContextProvider(**search_settings)

        # Create retrievers with different providers
        direct_retriever = EntityCompletionRetriever(
            extractor=DummyEntityExtractor(),
            context_provider=direct_provider,
            system_prompt_path="answer_simple_question.txt",
            user_prompt_path="context_for_question.txt",
        )

        summary_retriever = EntityCompletionRetriever(
            extractor=DummyEntityExtractor(),
            context_provider=summary_provider,
            system_prompt_path="answer_simple_question.txt",
            user_prompt_path="context_for_question.txt",
        )

        query = "What were the early contributions to quantum physics?"
        print("\nQuery:", query)

        # Try with direct triplets
        print("\n=== Direct Triplets ===")
        context = await direct_retriever.get_context(query)
        print("\nEntity Context:")
        print(context)

        result = await direct_retriever.get_completion(query)
        print("\nEntity Completion:")
        print(result)

        # Try with summarized triplets
        print("\n=== Summarized Triplets ===")
        context = await summary_retriever.get_context(query)
        print("\nEntity Context:")
        print(context)

        result = await summary_retriever.get_completion(query)
        print("\nEntity Completion:")
        print(result)

        # Compare with standard search
        print("\n=== Standard Search ===")
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=query
        )
        print(search_results)


if __name__ == "__main__":
    setup_logging(logging.ERROR)

    rebuild_kg = True
    retrieve = True
    steps_to_enable = {
        "prune_data": rebuild_kg,
        "prune_system": rebuild_kg,
        "add_text": rebuild_kg,
        "cognify": rebuild_kg,
        "retriever": retrieve,
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(steps_to_enable))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
