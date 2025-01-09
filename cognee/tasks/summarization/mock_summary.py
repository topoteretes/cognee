from cognee.shared.data_models import SummarizedCode, SummarizedClass, SummarizedFunction


def get_mock_summarized_code() -> SummarizedCode:
    return SummarizedCode(
        file_name="mock_file.py",
        high_level_summary="This is a mock high-level summary.",
        key_features=["Mock feature 1", "Mock feature 2"],
        imports=["mock_import1", "mock_import2"],
        constants=["MOCK_CONSTANT = 'mock_value'"],
        classes=[
            SummarizedClass(
                name="MockClass",
                description="This is a mock description of the MockClass.",
                methods=[
                    SummarizedFunction(
                        name="mock_method",
                        description="This is a description of the mock method.",
                        docstring="This is a mock method.",
                        inputs=["mock_input: str"],
                        outputs=["mock_output: str"],
                        decorators=None,
                    )
                ],
            )
        ],
        functions=[
            SummarizedFunction(
                name="mock_function",
                description="This is a description of the mock function.",
                docstring="This is a mock function.",
                inputs=["mock_input: str"],
                outputs=["mock_output: str"],
                decorators=None,
            )
        ],
        workflow_description="This is a mock workflow description.",
    )
