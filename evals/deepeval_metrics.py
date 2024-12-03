from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness_metric = GEval(
        name="Correctness",
        model="gpt-4o-mini",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        evaluation_steps=[
           "Determine whether the actual output is factually correct based on the expected output."    
        ]
    )
