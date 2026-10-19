"""Deterministic name-level checks shared by every place that matches text to ontology keys.

The resolver's fuzzy matcher (80% difflib cutoff) is tuned for single names —
``customers`` vs ``customer``. Query grounding, cognify's entity canonicalisation and
schema alignment all feed it multi-word strings as well, and a ratio over the whole
string will happily accept ``enterprise_customer_acme`` for ``EnterpriseCustomer``,
swallowing the token that named the individual. The helpers here are the guard rails
those callers share:

* :func:`multiword_match_is_sound` — every content token of a multi-word term must
  appear in the matched name, so a fuzzy hit cannot drag in an unrelated word.
* :func:`find_class_named_in` — the "contains the class name" rule: an extracted entity
  whose *name* ends its head noun phrase with an ontology class (``credit exposure to
  Acme`` → ``CreditExposure``) is grounded in that class even when its LLM-assigned
  *type* (``financial metric``) never fuzzy-matches anything.

No LLM, no I/O; every function is pure over strings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_MIN_TOKEN_LENGTH = 3

# Short function-word list: enough to keep "the", "our", "which" out of fuzzy matching
# without pulling in a stopword dependency. Words inside an n-gram are kept ("line of
# credit"), only n-gram boundaries and bare unigrams are filtered.
STOPWORDS = frozenset(
    """
    a an the and or but if then else of for to in on at by with from as into onto about
    over under between within without across after before during through this that these
    those there here what which who whom whose when where why how is are was were be been
    being am do does did done have has had having will would shall should can could may
    might must our your their its his her my me we you they them us it i he she not no yes
    all any each every some many much more most few less least own same other such than too
    very just also only ever never now new old up down out off so
    """.split()  # noqa: SIM905 — a word list reads better than 130 quoted literals
)

# Words that close an English head noun phrase: what precedes the first of them is the
# thing being named, what follows qualifies it ("credit exposure *to* acme corporation").
_PHRASE_BREAKERS = frozenset(
    """
    of to for in on at by with from as into onto about over under between within without
    across after before during through and or but versus vs & per via
    """.split()  # noqa: SIM905
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def normalize_key(name: str) -> str:
    """The resolver's key form: lowercase, spaces to underscores."""
    return name.lower().replace(" ", "_").strip()


def compact(name: str) -> str:
    """Strip every separator so ``credit_exposure`` and ``CreditExposure`` compare equal."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def tokens_of(name: str) -> list[str]:
    """Lowercased word tokens of a name in any casing (``camelCase`` is split too)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return TOKEN_PATTERN.findall(spaced.lower())


def singular(token: str) -> str:
    """A cheap singular form good enough for suffix comparison (never for display)."""
    if len(token) <= _MIN_TOKEN_LENGTH:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def multiword_match_is_sound(term: str, matched_name: str) -> bool:
    """Reject fuzzy matches where a multi-word term drags in an unrelated word.

    Every content word of a multi-word term must appear in the matched name (plural
    stripped); single words keep the resolver's own judgement.
    """
    tokens = [token for token in tokens_of(term) if token not in STOPWORDS]
    if len(tokens) <= 1:
        return True
    compact_name = compact(matched_name)
    return all(singular(token) in compact_name for token in tokens)


def head_phrase_tokens(name: str) -> list[str]:
    """Tokens of the head noun phrase: everything before the first phrase breaker.

    ``"credit exposure to acme corporation"`` → ``["credit", "exposure"]``;
    ``"acme corporation's credit exposure"`` → ``["acme", "corporation's", "credit",
    "exposure"]``. A name that *starts* with a breaker keeps it out of the head.
    """
    tokens = tokens_of(name)
    head: list[str] = []
    for token in tokens:
        if token in _PHRASE_BREAKERS:
            if head:
                break
            continue
        head.append(token)
    return head


def compact_class_index(class_keys: Iterable[str]) -> dict[str, str]:
    """``{compact form: key}`` for :func:`find_class_named_in`; build once per run."""
    compact_keys: dict[str, str] = {}
    for key in class_keys:
        compact_key = compact(key)
        if len(compact_key) >= _MIN_TOKEN_LENGTH:
            compact_keys.setdefault(compact_key, key)
    return compact_keys


def find_class_named_in(name: str, class_keys: Iterable[str] | Mapping[str, str]) -> str | None:
    """Return the class key that ends the head noun phrase of ``name``, if any.

    The class must be a *suffix* of the head phrase, token-aligned: the head noun is
    what the entity is. ``"customer feedback form"`` therefore does not become a
    ``Customer`` (its head is ``form``), while ``"acme customer"`` and ``"credit
    exposure to acme"`` do ground in ``Customer`` / ``CreditExposure``. The longest
    class wins; a class shorter than one content token of three characters is ignored.
    ``class_keys`` may be a prebuilt :func:`compact_class_index`.
    """
    head = head_phrase_tokens(name)
    if not head:
        return None
    if head[-1] in STOPWORDS:
        return None

    compact_keys = (
        class_keys if isinstance(class_keys, Mapping) else compact_class_index(class_keys)
    )

    best_key: str | None = None
    best_length = 0
    for start in range(len(head)):
        suffix = head[start:]
        if suffix[0] in STOPWORDS:
            continue
        for candidate in (
            "".join(suffix),
            "".join(suffix[:-1]) + singular(suffix[-1]),
        ):
            candidate = compact(candidate)
            key = compact_keys.get(candidate)
            if key is not None and len(candidate) > best_length:
                best_key, best_length = key, len(candidate)
    return best_key
