from .TextDocument import TextDocument


class CodeRepoDocument(TextDocument):
    """Document type for a code-repository manifest (system_metadata.source == "code_repo").

    Exists as the classification target the cognify CODE_REPO route keys on
    (modules/cognify/routing.py): the item's raw data is a small JSON manifest
    pointing at the original project directory, and the route runs one enola
    pass over that directory. Reading behaves like plain text (the manifest is
    JSON) for any consumer outside that route.
    """

    type: str = "code_repo"
    mime_type: str = "application/json"
