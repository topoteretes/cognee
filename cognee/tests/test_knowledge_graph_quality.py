import os
import asyncio
import cognee
import pathlib
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def test_knowledge_graph_quality_with_gpt4o():
    """
    Test that verifies all main concepts and entities from a specific document are found
    in the knowledge graph using the configured LLM model for entity extraction.

    This test addresses the issue where HotPotQA questions may not reflect diminishing
    quality of knowledge graph creation after data model changes.

    The model is configured via the LLM_MODEL environment variable.
    """

    # Ensure we have API key
    if not os.environ.get("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY must be set for this test")

    # Get model from environment variable
    current_model = os.environ.get("LLM_MODEL", "gpt-4o")
    print(f"Using model from environment: {current_model}")

    # Set up test directories
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_kg_quality")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_kg_quality")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean up before starting
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Get test document path
    test_document_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )

    # Expected entities and concepts from the NLP document
    expected_entities = [
        "Natural language processing",
        "NLP",
        "computer science",
        "information retrieval",
        "machine learning",
        "neural network",
        "speech recognition",
        "natural-language understanding",
        "natural-language generation",
        "theoretical linguistics",
        "text corpora",
        "speech corpora",
        "statistical approaches",
        "probabilistic approaches",
        "rule-based approaches",
        "documents",
        "language",
        "computers",
    ]

    expected_concepts = [
        "NLP is a subfield of computer science",
        "NLP is interdisciplinary",
        "NLP involves processing natural language datasets",
        "NLP uses machine learning approaches",
        "NLP borrows ideas from theoretical linguistics",
        "NLP can extract information from documents",
        "NLP can categorize and organize documents",
        "NLP involves speech recognition",
        "NLP involves natural-language understanding",
        "NLP involves natural-language generation",
        "computers can understand document contents",
        "neural networks are used in NLP",
        "statistical approaches are used in NLP",
    ]

    print("=" * 80)
    print("KNOWLEDGE GRAPH QUALITY TEST WITH GPT-4o")
    print("=" * 80)
    print(f"Using model: {os.environ.get('LLM_MODEL', 'gpt-4o')}")
    print(f"Test document: {test_document_path}")
    print()

    # Add and process the document
    print("Adding document to cognee...")
    await cognee.add([test_document_path], dataset_name="NLP_TEST")

    user = await get_default_user()

    print("Processing document with cognify...")
    await cognee.cognify(["NLP_TEST"], user=user)
    print("Document processing completed.")
    print()

    # Test different search types to find entities and concepts
    search_types_to_test = [
        (SearchType.INSIGHTS, "Get entity relationships and connections"),
        (SearchType.GRAPH_COMPLETION, "Natural language completion with graph context"),
        (SearchType.CHUNKS, "Find relevant document chunks"),
        (SearchType.SUMMARIES, "Get content summaries"),
    ]

    all_found_results = {}

    for search_type, description in search_types_to_test:
        print(f"Testing {search_type.value} search - {description}")
        print("-" * 60)

        # Search for entities
        entity_results = await cognee.search(
            query_type=search_type,
            query_text="What are the main entities, concepts, and terms mentioned in this document?",
            user=user,
            top_k=20,
        )

        # Search for relationships
        relationship_results = await cognee.search(
            query_type=search_type,
            query_text="What are the key relationships and connections between concepts in this document?",
            user=user,
            top_k=20,
        )

        all_found_results[search_type.value] = {
            "entities": entity_results,
            "relationships": relationship_results,
        }

        print(f"Entity search results ({len(entity_results)} items):")
        for i, result in enumerate(entity_results[:3]):  # Show first 3 results
            print(f"  {i + 1}. {result}")

        print(f"Relationship search results ({len(relationship_results)} items):")
        for i, result in enumerate(relationship_results[:3]):  # Show first 3 results
            print(f"  {i + 1}. {result}")
        print()

    # Analyze results and check for expected entities and concepts
    print("ANALYSIS: Expected vs Found")
    print("=" * 80)

    # Combine all results into a single text for analysis
    all_results_text = ""
    for search_type, results in all_found_results.items():
        for result_type, result_list in results.items():
            all_results_text += f" {' '.join(str(r) for r in result_list)}"

    all_results_text = all_results_text.lower()

    print("ENTITY ANALYSIS:")
    print("-" * 40)
    found_entities = []
    missing_entities = []

    for entity in expected_entities:
        entity_lower = entity.lower()
        # Check if entity or its variations are found
        if (
            entity_lower in all_results_text
            or entity_lower.replace("-", " ") in all_results_text
            or entity_lower.replace(" ", "-") in all_results_text
        ):
            found_entities.append(entity)
            print(f"✓ FOUND: {entity}")
        else:
            missing_entities.append(entity)
            print(f"✗ MISSING: {entity}")

    print()
    print("CONCEPT ANALYSIS:")
    print("-" * 40)
    found_concepts = []
    missing_concepts = []

    for concept in expected_concepts:
        concept_lower = concept.lower()
        # Check if key parts of the concept are found
        concept_words = concept_lower.split()
        key_words = [
            word
            for word in concept_words
            if len(word) > 2
            and word not in ["the", "and", "are", "can", "involves", "uses", "from"]
        ]

        if len(key_words) > 0:
            found_key_words = sum(1 for word in key_words if word in all_results_text)
            coverage = found_key_words / len(key_words)

            if coverage >= 0.6:  # At least 60% of key words found
                found_concepts.append(concept)
                print(f"✓ FOUND: {concept} (coverage: {coverage:.1%})")
            else:
                missing_concepts.append(concept)
                print(f"✗ MISSING: {concept} (coverage: {coverage:.1%})")
        else:
            missing_concepts.append(concept)
            print(f"✗ MISSING: {concept} (no key words)")

    print()
    print("SUMMARY:")
    print("=" * 40)
    print(f"Expected entities: {len(expected_entities)}")
    print(f"Found entities: {len(found_entities)}")
    print(f"Missing entities: {len(missing_entities)}")
    print(f"Entity coverage: {len(found_entities) / len(expected_entities):.1%}")
    print()
    print(f"Expected concepts: {len(expected_concepts)}")
    print(f"Found concepts: {len(found_concepts)}")
    print(f"Missing concepts: {len(missing_concepts)}")
    print(f"Concept coverage: {len(found_concepts) / len(expected_concepts):.1%}")
    print()

    # Test assertions
    entity_coverage = len(found_entities) / len(expected_entities)
    concept_coverage = len(found_concepts) / len(expected_concepts)

    print("QUALITY ASSESSMENT:")
    print("-" * 40)
    print(f"Model used: {current_model}")
    print()

    # Adjust quality thresholds based on model capability
    if current_model == "gpt-4o":
        min_entity_coverage = 0.70  # At least 70% of entities should be found
        min_concept_coverage = 0.60  # At least 60% of concepts should be found
    elif current_model == "gpt-4o-mini":
        min_entity_coverage = 0.65  # Slightly lower for mini model
        min_concept_coverage = 0.55  # Slightly lower for mini model
    elif current_model == "gpt-4-turbo":
        min_entity_coverage = 0.68  # Good performance expected
        min_concept_coverage = 0.58  # Good performance expected
    else:  # gpt-3.5-turbo or other models
        min_entity_coverage = 0.60  # Lower threshold for older models
        min_concept_coverage = 0.50  # Lower threshold for older models

    if entity_coverage >= min_entity_coverage:
        print(
            f"✓ PASS: Entity coverage ({entity_coverage:.1%}) meets minimum requirement ({min_entity_coverage:.1%})"
        )
    else:
        print(
            f"✗ FAIL: Entity coverage ({entity_coverage:.1%}) below minimum requirement ({min_entity_coverage:.1%})"
        )

    if concept_coverage >= min_concept_coverage:
        print(
            f"✓ PASS: Concept coverage ({concept_coverage:.1%}) meets minimum requirement ({min_concept_coverage:.1%})"
        )
    else:
        print(
            f"✗ FAIL: Concept coverage ({concept_coverage:.1%}) below minimum requirement ({min_concept_coverage:.1%})"
        )

    overall_quality = (entity_coverage + concept_coverage) / 2
    print(f"Overall quality score: {overall_quality:.1%}")

    # Assert that we have acceptable quality
    assert entity_coverage >= min_entity_coverage, (
        f"Entity coverage {entity_coverage:.1%} below minimum {min_entity_coverage:.1%}"
    )
    assert concept_coverage >= min_concept_coverage, (
        f"Concept coverage {concept_coverage:.1%} below minimum {min_concept_coverage:.1%}"
    )

    print()
    print("=" * 80)
    print("KNOWLEDGE GRAPH QUALITY TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return {
        "entity_coverage": entity_coverage,
        "concept_coverage": concept_coverage,
        "overall_quality": overall_quality,
        "found_entities": found_entities,
        "missing_entities": missing_entities,
        "found_concepts": found_concepts,
        "missing_concepts": missing_concepts,
    }


if __name__ == "__main__":
    asyncio.run(test_knowledge_graph_quality_with_gpt4o())
