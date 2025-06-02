from .config import get_relational_config
from .create_relational_engine import create_relational_engine


def get_relational_engine():
    """
    Create and return a relational engine instance.

    This function retrieves the relational configuration and then uses it to create a
    relational engine using provided settings. The behavior of this function depends on the
    successful retrieval and conversion of the relational configuration into the appropriate
    format for the engine creation.

    Returns:
    --------

        A relational engine instance configured according to the retrieved settings.
    """
    relational_config = get_relational_config()

    return create_relational_engine(**relational_config.to_dict())
