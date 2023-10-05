from deepeval.metrics.overall_score import OverallScoreMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test, run_test

import uuid


def retrieve_test_cases():
    """Retrieve test cases from a database or a file."""
    pass


def check_params(chunk_size, chunk_overlap, chunk_strategy, loader_strategy, query, context, metadata):
    """Check parameters for test case runs and set defaults if necessary."""
    pass


def run_load(test_id, document, **kwargs):
    """Run load for the given test_id and document with other parameters."""
    pass


def compare_output(output, expected_output):
    """Compare the output against the expected output."""
    pass


def generate_param_variants(base_params):
    """Generate parameter variants for testing."""
    params_variants = [
                          {'chunk_size': base_params['chunk_size'] + i} for i in range(1, 4)
                      ] + [
                          {'chunk_overlap': base_params['chunk_overlap'] + i} for i in range(1, 4)
                      ]
    # Add more parameter variations here as needed
    return params_variants


def run_tests_with_variants(document, base_params, param_variants, expected_output):
    """Run tests with various parameter variants and validate the output."""
    for variant in param_variants:
        test_id = str(uuid.uuid4())  # Set new test id
        updated_params = {**base_params, **variant}  # Update parameters
        output = run_load(test_id, document, **updated_params)  # Run load with varied parameters
        compare_output(output, expected_output)  # Validate output


def run_rag_tests(document, chunk_size, chunk_overlap, chunk_strategy, loader_strategy, query, output, expected_output,
                  context, metadata):
    """Run RAG tests with various scenarios and parameter variants."""
    test_cases = retrieve_test_cases()  # Retrieve test cases

    # Check and set parameters
    base_params = check_params(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        loader_strategy=loader_strategy,
        query=query,
        context=context,
        metadata=metadata
    )

    # Set test id and run initial load test
    test_id = str(uuid.uuid4())
    output = run_load(test_id, document, **base_params)
    compare_output(output, expected_output)

    # Generate parameter variants for further tests
    param_variants = generate_param_variants(base_params)

    # Run tests with varied parameters for the single document
    run_tests_with_variants(document, base_params, param_variants, expected_output)

    # Assuming two documents are concatenated and treated as one
    combined_document = document + document

    # Run initial load test for combined document
    output = run_load(test_id, combined_document, **base_params)
    compare_output(output, expected_output)

    # Run tests with varied parameters for the combined document
    run_tests_with_variants(combined_document, base_params, param_variants, expected_output)


def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    test_case = LLMTestCase(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    metric = OverallScoreMetric()
    # if you want to make sure that the test returns an error
    assert_test(test_case, metrics=[metric])

    # If you want to run the test
    test_result = run_test(test_case, metrics=[metric])
    # You can also inspect the test result class
    print(test_result)
    print(test_result)