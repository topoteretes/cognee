from urllib.parse import urlparse


def get_scheme_with_separator(url: str) -> str:
    """
    from a specified URL, return the full scheme (e.g. 's3://')

    Args:
        url: a standard URL string.

    Returns:
        return the full scheme from the URL, and end with "://".
        if the URL has no scheme, return "://".
    """
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    scheme_with_separator = f"{scheme}://"
    return scheme_with_separator
