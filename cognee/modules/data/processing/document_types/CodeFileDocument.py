from .TextDocument import TextDocument


class CodeFileDocument(TextDocument):
    """Document type for a supported code file (system_metadata.source == "code").

    Exists as the classification target the cognify CODE route keys on
    (modules/cognify/routing.py): such items run the deterministic enola code
    graph pipeline instead of LLM chunk extraction. Reading behaves like a
    plain text document — code is text — for any consumer outside that route.
    """

    type: str = "code_file"
    mime_type: str = "text/plain"
