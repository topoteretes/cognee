from cognee.infrastructure.databases.exceptions.exceptions import (
    EntityNotFoundError,
    NodesetFilterNotSupportedError,
)


def test_entity_not_found_error_initializes_base_exception_state():
    error = EntityNotFoundError()

    assert error.message == "The requested entity does not exist."
    assert error.name == "EntityNotFoundError"
    assert error.status_code == 404
    assert error.args == (error.message, error.name)


def test_nodeset_filter_not_supported_error_initializes_base_exception_state():
    error = NodesetFilterNotSupportedError()

    assert error.message == "The nodeset filter is not supported in the current graph database."
    assert error.name == "NodeSetFilterNotSupportedError"
    assert error.status_code == 404
    assert error.args == (error.message, error.name)