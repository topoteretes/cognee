import os
import cognee
import pathlib

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user, create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets


async def test_knowledge_graph_quality_with_gpt4o():
    """
    Test that verifies all main concepts and entities from a specific document are found
    in the knowledge graph using GPT-4o model for high-quality entity extraction.

    This test addresses the issue where HotPotQA questions may not reflect diminishing
    quality of knowledge graph creation after data model changes.
    """

    # Configure GPT-4o for best quality
    os.environ["LLM_MODEL"] = "gpt-4o"
    cognee.config.set_llm_model("gpt-4o")

    # Ensure we have API key
    if not os.environ.get("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY must be set for this test")

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

    # We expect high coverage with GPT-4o
    min_entity_coverage = 0.70  # At least 70% of entities should be found
    min_concept_coverage = 0.60  # At least 60% of concepts should be found

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


logger = get_logger()


async def main():
    # Enable permissions feature
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    # Clean up test directories before starting
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_permissions")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_permissions")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )

    # Add document for default user
    await cognee.add([explanation_file_path], dataset_name="NLP")
    default_user = await get_default_user()

    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
    Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
    The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
    Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
    In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
    """

    # Add document for test user
    test_user = await create_user("user@example.com", "example")
    await cognee.add([text], dataset_name="QUANTUM", user=test_user)

    await cognee.cognify(["NLP"], user=default_user)
    await cognee.cognify(["QUANTUM"], user=test_user)

    # Check if default_user can only see information from the NLP dataset
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=default_user,
    )
    assert len(search_results) == 1, "The search results list lenght is not one."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")
    assert search_results[0]["dataset_name"] == "NLP", (
        f"Dict must contain dataset name 'NLP': {search_results[0]}"
    )

    # Check if test_user can only see information from the QUANTUM dataset
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=test_user,
    )
    assert len(search_results) == 1, "The search results list lenght is not one."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")
    assert search_results[0]["dataset_name"] == "QUANTUM", (
        f"Dict must contain dataset name 'QUANTUM': {search_results[0]}"
    )

    # Try to add document with default_user to test_users dataset (test write permission enforcement)
    test_user_dataset_id = search_results[0]["dataset_id"]
    add_error = False
    try:
        await cognee.add(
            [explanation_file_path],
            dataset_name="QUANTUM",
            dataset_id=test_user_dataset_id,
            user=default_user,
        )
    except PermissionDeniedError:
        add_error = True
    assert add_error, "PermissionDeniedError was not raised during add as expected"

    # Try to cognify with default_user the test_users dataset (test write permission enforcement)
    cognify_error = False
    try:
        await cognee.cognify(datasets=[test_user_dataset_id], user=default_user)
    except PermissionDeniedError:
        cognify_error = True
    assert cognify_error, "PermissionDeniedError was not raised during cognify as expected"

    # Try to add permission for a dataset default_user does not have share permission for
    give_permission_error = False
    try:
        await authorized_give_permission_on_datasets(
            default_user.id,
            [test_user_dataset_id],
            "write",
            default_user.id,
        )
    except PermissionDeniedError:
        give_permission_error = True
    assert give_permission_error, (
        "PermissionDeniedError was not raised during assignment of permission as expected"
    )

    # Actually give permission to default_user to write on test_users dataset
    await authorized_give_permission_on_datasets(
        default_user.id,
        [test_user_dataset_id],
        "write",
        test_user.id,
    )

    # Add new data to test_users dataset from default_user
    await cognee.add(
        [explanation_file_path],
        dataset_name="QUANTUM",
        dataset_id=test_user_dataset_id,
        user=default_user,
    )
    await cognee.cognify(datasets=[test_user_dataset_id], user=default_user)

    # Actually give permission to default_user to read on test_users dataset
    await authorized_give_permission_on_datasets(
        default_user.id,
        [test_user_dataset_id],
        "read",
        test_user.id,
    )

    # Check if default_user can see from test_users datasets now
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=default_user,
        dataset_ids=[test_user_dataset_id],
    )
    assert len(search_results) == 1, "The search results list length is not one."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    assert search_results[0]["dataset_name"] == "QUANTUM", (
        f"Dict must contain dataset name 'QUANTUM': {search_results[0]}"
    )

    # Check if default_user can only see information from both datasets now
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=default_user,
    )
    assert len(search_results) == 2, "The search results list length is not two."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    # Try deleting data from test_user dataset with default_user without delete permission
    delete_error = False
    try:
        await cognee.delete([text], dataset_id=test_user_dataset_id, user=default_user)
    except PermissionDeniedError:
        delete_error = True

    assert delete_error, "PermissionDeniedError was not raised during delete operation as expected"

    # Try deleting data from test_user dataset with test_user
    await cognee.delete([text], dataset_id=test_user_dataset_id, user=test_user)

    # Actually give permission to default_user to delete data for test_users dataset
    await authorized_give_permission_on_datasets(
        default_user.id,
        [test_user_dataset_id],
        "delete",
        test_user.id,
    )

    # Try deleting data from test_user dataset with default_user after getting delete permission
    await cognee.delete([explanation_file_path], dataset_id=test_user_dataset_id, user=default_user)


async def main_quality_test():
    """Main function to run the knowledge graph quality test"""
    await test_knowledge_graph_quality_with_gpt4o()


if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quality":
        print("Running Knowledge Graph Quality Test...")
        asyncio.run(main_quality_test())
    else:
        print("Running Permissions Test...")
        asyncio.run(main())
