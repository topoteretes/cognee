"""Deterministic name-level checks shared by every place that matches text to ontology keys.

The resolver's fuzzy matcher (80% difflib cutoff) is tuned for single names —
``customers`` vs ``customer``. Cognify's entity canonicalisation feeds it multi-word
strings as well, and a ratio over the whole string will happily accept ``enterprise_customer_acme`` for ``EnterpriseCustomer``,
swallowing the token that named the individual. The helpers here are the guard rails
those callers share:

* :func:`multiword_match_is_sound` — every content token of a multi-word term must
  appear in the matched name, so a fuzzy hit cannot drag in an unrelated word.
* :func:`lookup_words` — the exact lookup: a list of words, as written or with the last
  word singular, against a :func:`compact_key_index`. Recall grounding and the rule
  below use it.
* :func:`find_class_named_in` — the "contains the class name" rule: an extracted entity
  whose *name* ends its head noun phrase with an ontology class (``credit exposure to
  Acme`` → ``CreditExposure``) is grounded in that class even when its LLM-assigned
  *type* (``financial metric``) never fuzzy-matches anything.

No LLM, no I/O; every function is pure over strings (``compact_key_index`` reads the
resolver's in-memory lookup table).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver

# Tokens this short are never singularized ("bus", "gas", "its").
_MIN_SINGULAR_LENGTH = 3

# Short function-word list, without a stopword dependency. They are not content words
# of a fuzzy match (``multiword_match_is_sound``), and one alone never matches a key
# ("will", "may" are not the classes ``Will`` / ``May``). Inside or at the end of a
# longer run they stay: an exact key decides ("line of credit", "new york").
_STOPWORDS = frozenset(
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

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def compact(name: str) -> str:
    """Strip every separator so ``credit_exposure`` and ``CreditExposure`` compare equal."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def tokens_of(name: str) -> list[str]:
    """Lowercased word tokens of a name in any casing (``camelCase`` is split too)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return _TOKEN_PATTERN.findall(spaced.lower())


def singular(token: str) -> str:
    """A cheap singular form good enough for suffix comparison (never for display)."""
    if len(token) <= _MIN_SINGULAR_LENGTH:
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
    tokens = [token for token in tokens_of(term) if token not in _STOPWORDS]
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


def compact_key_index(resolver: BaseOntologyResolver, category: str) -> dict[str, str]:
    """``{compact form: resolver key}`` for one lookup category; build once per use.

    ``lookup`` is read the way ``ensure_ontology_usable_in_strict_mode`` reads it: a
    custom resolver without a ``lookup`` dict gets an empty index.
    """
    lookup = getattr(resolver, "lookup", None)
    keys = lookup.get(category) if isinstance(lookup, dict) else None
    if not isinstance(keys, dict):
        return {}
    index: dict[str, str] = {}
    for key in keys:
        compact_key = compact(key)
        if compact_key:
            index.setdefault(compact_key, key)
    return index


def lookup_words(words: list[str], index: Mapping[str, str]) -> str | None:
    """The key of ``index`` (a :func:`compact_key_index`) that ``words`` name, or None.

    The words are compared in compact form as written and with the last word singular,
    so "credit exposures" finds ``creditexposure``. A lone stopword never matches.
    """
    if not words or (len(words) == 1 and words[0] in _STOPWORDS):
        return None
    for form in ("".join(words), "".join(words[:-1]) + singular(words[-1])):
        key = index.get(compact(form))
        if key is not None:
            return key
    return None


def find_class_named_in(name: str, class_index: Mapping[str, str]) -> str | None:
    """Return the class key that ends the head noun phrase of ``name``, if any.

    The class must be a *suffix* of the head phrase, token-aligned: the head noun is
    what the entity is. ``"customer feedback form"`` therefore does not become a
    ``Customer`` (its head is ``form``), while ``"acme customer"`` and ``"credit
    exposure to acme"`` do ground in ``Customer`` / ``CreditExposure``. The longest
    class wins. The head ends at the first phrase breaker (``of``, ``to``, ``for``, …), so
    a class whose name contains one (``LineOfCredit``) is never found inside a name.
    ``class_index`` is the classes' :func:`compact_key_index`.
    """
    head = head_phrase_tokens(name)
    for start in range(len(head)):
        key = lookup_words(head[start:], class_index)
        if key is not None:
            return key
    return None
