import os

from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
from cognee.infrastructure.databases.graph.config import get_graph_context_config


VECTOR_DBS_WITH_MULTI_USER_SUPPORT = ["lancedb", "falkor"]
GRAPH_DBS_WITH_MULTI_USER_SUPPORT = ["kuzu", "falkor"]


def is_multi_user_support_possible():
    graph_db_config = get_graph_context_config()
    vector_db_config = get_vectordb_context_config()
    return (
        graph_db_config["graph_database_provider"] in GRAPH_DBS_WITH_MULTI_USER_SUPPORT
        and vector_db_config["vector_db_provider"] in VECTOR_DBS_WITH_MULTI_USER_SUPPORT
    )


def is_backend_access_control_enabled():
    backend_access_control = os.environ.get("ENABLE_BACKEND_ACCESS_CONTROL", None)
    if backend_access_control is None:
        # If backend access control is not defined in environment variables,
        # enable it by default if graph and vector DBs can support it, otherwise disable it
        return is_multi_user_support_possible()
    elif backend_access_control.lower() == "true":
        # If enabled, ensure that the current graph and vector DBs can support it
        multi_user_support = is_multi_user_support_possible()
        if not multi_user_support:
            raise EnvironmentError(
                "ENABLE_BACKEND_ACCESS_CONTROL is set to true but the current graph and/or vector databases do not support multi-user access control. Please use supported databases or disable backend access control."
            )
        return True
    return False
