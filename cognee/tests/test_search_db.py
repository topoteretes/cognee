import os
import pathlib

from dns.e164 import query

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.engine.models import NodeSet
from collections import Counter

logger = get_logger()


async def main():
    # This test runs for multiple db settings, to run this locally set the corresponding db envs
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "test_dataset"

    text_1 = """Germany is located in europe right next to the Netherlands"""
    await cognee.add(text_1, dataset_name)

    text = """A quantum computer is a computer that takes advantage of quantum mechanical phenomena.
    At small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior, specifically quantum superposition and entanglement, using specialized hardware that supports the preparation and manipulation of quantum states.
    Classical physics cannot explain the operation of these quantum devices, and a scalable quantum computer could perform some calculations exponentially faster (with respect to input size scaling) than any modern "classical" computer. In particular, a large-scale quantum computer could break widely used encryption schemes and aid physicists in performing physical simulations; however, the current state of the technology is largely experimental and impractical, with several obstacles to useful applications. Moreover, scalable quantum computers do not hold promise for many practical tasks, and for many important tasks quantum speedups are proven impossible.
    The basic unit of information in quantum computing is the qubit, similar to the bit in traditional digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis" states. When measuring a qubit, the result is a probabilistic output of a classical bit, therefore making quantum computers nondeterministic in general. If a quantum computer manipulates the qubit in a particular way, wave interference effects can amplify the desired measurement results. The design of quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently and quickly.
    Physically engineering high-quality qubits has proven challenging. If a physical qubit is not sufficiently isolated from its environment, it suffers from quantum decoherence, introducing noise into calculations. Paradoxically, perfectly isolating qubits is also undesirable because quantum computations typically need to initialize qubits, perform controlled qubit interactions, and measure the resulting quantum states. Each of those operations introduces errors and suffers from noise, and such inaccuracies accumulate.
    In principle, a non-quantum (classical) computer can solve the same computational problems as a quantum computer, given enough time. Quantum advantage comes in the form of time complexity rather than computability, and quantum complexity theory shows that some quantum algorithms for carefully selected tasks require exponentially fewer computational steps than the best known non-quantum algorithms. Such tasks can in theory be solved on a large-scale quantum computer whereas classical computers would not finish computations in any reasonable amount of time. However, quantum speedup is not universal or even typical across computational tasks, since basic tasks such as sorting are proven to not allow any asymptotic quantum speedup. Claims of quantum supremacy have drawn significant attention to the discipline, but are demonstrated on contrived tasks, while near-term practical use cases remain limited.
    """

    await cognee.add([text], dataset_name)

    await cognee.cognify([dataset_name])

    context_gk, _ = await GraphCompletionRetriever().get_context(
        query="Next to which country is Germany located?"
    )
    context_gk_cot, _ = await GraphCompletionCotRetriever().get_context(
        query="Next to which country is Germany located?"
    )
    context_gk_ext, _ = await GraphCompletionContextExtensionRetriever().get_context(
        query="Next to which country is Germany located?"
    )
    context_gk_sum, _ = await GraphSummaryCompletionRetriever().get_context(
        query="Next to which country is Germany located?"
    )

    for name, context in [
        ("GraphCompletionRetriever", context_gk),
        ("GraphCompletionCotRetriever", context_gk_cot),
        ("GraphCompletionContextExtensionRetriever", context_gk_ext),
        ("GraphSummaryCompletionRetriever", context_gk_sum),
    ]:
        assert isinstance(context, str), f"{name}: Context should be a string"
        assert context.strip(), f"{name}: Context should not be empty"
        lower = context.lower()
        assert "germany" in lower or "netherlands" in lower, (
            f"{name}: Context did not contain 'germany' or 'netherlands'; got: {context!r}"
        )

    triplets_gk = await GraphCompletionRetriever().get_triplets(
        query="Next to which country is Germany located?"
    )
    triplets_gk_cot = await GraphCompletionCotRetriever().get_triplets(
        query="Next to which country is Germany located?"
    )
    triplets_gk_ext = await GraphCompletionContextExtensionRetriever().get_triplets(
        query="Next to which country is Germany located?"
    )
    triplets_gk_sum = await GraphSummaryCompletionRetriever().get_triplets(
        query="Next to which country is Germany located?"
    )

    for name, triplets in [
        ("GraphCompletionRetriever", triplets_gk),
        ("GraphCompletionCotRetriever", triplets_gk_cot),
        ("GraphCompletionContextExtensionRetriever", triplets_gk_ext),
        ("GraphSummaryCompletionRetriever", triplets_gk_sum),
    ]:
        assert isinstance(triplets, list), f"{name}: Triplets should be a list"
        assert triplets, f"{name}: Triplets list should not be empty"
        for edge in triplets:
            assert isinstance(edge, Edge), f"{name}: Elements should be Edge instances"
            distance = edge.attributes.get("vector_distance")
            node1_distance = edge.node1.attributes.get("vector_distance")
            node2_distance = edge.node2.attributes.get("vector_distance")
            assert isinstance(distance, float), (
                f"{name}: vector_distance should be float, got {type(distance)}"
            )
            assert 0 <= distance <= 1, (
                f"{name}: edge vector_distance {distance} out of [0,1], this shouldn't happen"
            )
            assert 0 <= node1_distance <= 1, (
                f"{name}: node_1 vector_distance {distance} out of [0,1], this shouldn't happen"
            )
            assert 0 <= node2_distance <= 1, (
                f"{name}: node_2 vector_distance {distance} out of [0,1], this shouldn't happen"
            )

    completion_gk = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where is germany located, next to which country?",
        save_interaction=True,
    )
    completion_cot = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_COT,
        query_text="What is the country next to germany??",
        save_interaction=True,
    )
    completion_ext = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        query_text="What is the name of the country next to germany",
        save_interaction=True,
    )

    await cognee.search(
        query_type=SearchType.FEEDBACK, query_text="This was not the best answer", last_k=1
    )

    completion_sum = await cognee.search(
        query_type=SearchType.GRAPH_SUMMARY_COMPLETION,
        query_text="Next to which country is Germany located?",
        save_interaction=True,
    )

    await cognee.search(
        query_type=SearchType.FEEDBACK,
        query_text="This answer was great",
        last_k=1,
    )

    for name, completion in [
        ("GRAPH_COMPLETION", completion_gk),
        ("GRAPH_COMPLETION_COT", completion_cot),
        ("GRAPH_COMPLETION_CONTEXT_EXTENSION", completion_ext),
        ("GRAPH_SUMMARY_COMPLETION", completion_sum),
    ]:
        assert isinstance(completion, list), f"{name}: should return a list"
        assert len(completion) == 1, f"{name}: expected single-element list, got {len(completion)}"
        text = completion[0]
        assert isinstance(text, str), f"{name}: element should be a string"
        assert text.strip(), f"{name}: string should not be empty"
        assert "netherlands" in text.lower(), (
            f"{name}: expected 'netherlands' in result, got: {text!r}"
        )

    graph_engine = await get_graph_engine()
    graph = await graph_engine.get_graph_data()

    type_counts = Counter(node_data[1].get("type", {}) for node_data in graph[0])

    edge_type_counts = Counter(edge_type[2] for edge_type in graph[1])

    # Assert there are exactly 4 CogneeUserInteraction nodes.
    assert type_counts.get("CogneeUserInteraction", 0) == 4, (
        f"Expected exactly four DCogneeUserInteraction nodes, but found {type_counts.get('CogneeUserInteraction', 0)}"
    )

    # Assert there is exactly two CogneeUserFeedback nodes.
    assert type_counts.get("CogneeUserFeedback", 0) == 2, (
        f"Expected exactly two CogneeUserFeedback nodes, but found {type_counts.get('CogneeUserFeedback', 0)}"
    )

    # Assert there is exactly two NodeSet.
    assert type_counts.get("NodeSet", 0) == 2, (
        f"Expected exactly two NodeSet nodes, but found {type_counts.get('NodeSet', 0)}"
    )

    # Assert that there are at least 10 'used_graph_element_to_answer' edges.
    assert edge_type_counts.get("used_graph_element_to_answer", 0) >= 10, (
        f"Expected at least ten 'used_graph_element_to_answer' edges, but found {edge_type_counts.get('used_graph_element_to_answer', 0)}"
    )

    # Assert that there are exactly 2 'gives_feedback_to' edges.
    assert edge_type_counts.get("gives_feedback_to", 0) == 2, (
        f"Expected exactly two 'gives_feedback_to' edges, but found {edge_type_counts.get('gives_feedback_to', 0)}"
    )

    # Assert that there are at least 6 'belongs_to_set' edges.
    assert edge_type_counts.get("belongs_to_set", 0) == 6, (
        f"Expected at least six 'belongs_to_set' edges, but found {edge_type_counts.get('belongs_to_set', 0)}"
    )

    nodes = graph[0]

    required_fields_user_interaction = {"question", "answer", "context"}
    required_fields_feedback = {"feedback", "sentiment"}

    for node_id, data in nodes:
        if data.get("type") == "CogneeUserInteraction":
            assert required_fields_user_interaction.issubset(data.keys()), (
                f"Node {node_id} is missing fields: {required_fields_user_interaction - set(data.keys())}"
            )

            for field in required_fields_user_interaction:
                value = data[field]
                assert isinstance(value, str) and value.strip(), (
                    f"Node {node_id} has invalid value for '{field}': {value!r}"
                )

        if data.get("type") == "CogneeUserFeedback":
            assert required_fields_feedback.issubset(data.keys()), (
                f"Node {node_id} is missing fields: {required_fields_feedback - set(data.keys())}"
            )

            for field in required_fields_feedback:
                value = data[field]
                assert isinstance(value, str) and value.strip(), (
                    f"Node {node_id} has invalid value for '{field}': {value!r}"
                )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(text_1, dataset_name)

    await cognee.add([text], dataset_name)

    await cognee.cognify([dataset_name])

    await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Next to which country is Germany located?",
        save_interaction=True,
    )

    await cognee.search(
        query_type=SearchType.FEEDBACK,
        query_text="This was the best answer I've ever seen",
        last_k=1,
    )

    await cognee.search(
        query_type=SearchType.FEEDBACK,
        query_text="Wow the correctness of this answer blows my mind",
        last_k=1,
    )

    graph = await graph_engine.get_graph_data()

    edges = graph[1]

    for from_node, to_node, relationship_name, properties in edges:
        if relationship_name == "used_graph_element_to_answer":
            assert properties["feedback_weight"] >= 6, (
                "Feedback weight calculation is not correct, it should be more then 6."
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
