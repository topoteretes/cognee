from enum import Enum


class SearchStatus(str, Enum):
    """Why a completion search's ``completion`` is what it is.

    A completion retriever never sends an empty retrieval context to the LLM: the
    only possible output is a phantom "no context provided" deflection that callers
    cannot tell apart from a real answer (SDK-270 / gh #3728). When the LLM is
    skipped the ``completion`` stays empty, so ``if results:`` keeps meaning "memory
    answered", and this status says *why* it is empty. It is set per dataset, so one
    empty dataset in a multi-dataset search never hides its siblings' answers.
    """

    # The completion was generated (or the search type has no completion step).
    OK = "ok"
    # The graph holds data but retrieval matched nothing for this query, so no LLM
    # call was made. A normal miss on a healthy system.
    NO_CONTEXT = "no_context"
    # The knowledge graph is empty for this dataset (fresh install, add() without
    # cognify(), or memory dropped): retrieval and the LLM call were both skipped.
    GRAPH_EMPTY = "graph_empty"
