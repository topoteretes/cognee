from contextvars import ContextVar

# Note: ContextVar allows us to use different graph db configurations in Cognee
#       for different async tasks, threads and processes
vector_db_config = ContextVar("vector_db_config", default=None)
graph_db_config = ContextVar("graph_db_config", default=None)
