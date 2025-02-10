from cognee.shared.utils import start_visualization_server


def visualization_server(port):
    """
    Start a visualization server on the specified port.

    Args:
        port (int): The port number to run the server on

    Returns:
        callable: A shutdown function that can be called to stop the server

    Raises:
        ValueError: If port is not a valid port number
    """
    return start_visualization_server(port=port)
