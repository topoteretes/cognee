import math

import pytest

from cognee.modules.retrieval.code_search_index import (
    STOPWORDS,
    CodeSearchIndex,
    query_terms,
    split_identifier,
    stem,
    tokenize,
)


@pytest.mark.parametrize(
    "identifier, parts",
    [
        ("validateToken", ["validate", "Token"]),
        ("HTTPServer", ["HTTP", "Server"]),
        ("getHTTPResponse", ["get", "HTTP", "Response"]),
        ("IOError", ["IO", "Error"]),
        ("sha256Hash", ["sha256", "Hash"]),
        ("parse", ["parse"]),
        ("URL", ["URL"]),
    ],
)
def test_identifiers_split_at_camel_case_and_acronym_boundaries(identifier, parts):
    assert split_identifier(identifier) == parts


def test_camel_and_snake_case_names_yield_the_same_terms():
    assert tokenize("validateToken") == ["valid", "token"]
    assert tokenize("validate_token") == ["valid", "token"]
    assert tokenize("auth.validate-token:v2") == ["auth", "valid", "token", "v2"]
    assert tokenize("app/auth/tokens.py") == ["app", "auth", "token", "py"]
    assert tokenize("HTTPServer") == ["http", "serv"]


def test_stopwords_question_filler_and_short_tokens_are_dropped():
    assert tokenize("How does checkout work in this repo?") == ["checkout"]
    assert tokenize("Which code deals with a codebase repository?") == []
    assert tokenize("x y 7 ab") == ["ab"]
    assert {"how", "what", "work", "code", "repo"} <= STOPWORDS
    assert tokenize(None) == [] and tokenize("") == []


@pytest.mark.parametrize(
    "forms",
    [
        "validate validates validated validating validation validator",
        "authenticate authentication authenticator",
        "authorize authorization authorized",
        "process processes processing processed processor",
        "handle handles handled handling handler",
        "token tokens",
        "query queries",
    ],
)
def test_word_forms_share_one_stem(forms):
    assert len({stem(word) for word in forms.split()}) == 1


def test_stemmer_keeps_short_stems_and_words_ending_in_ss_us_is():
    assert stem("class") == "class"
    assert stem("status") == "status"
    assert stem("analysis") == "analysis"
    assert stem("user") == "user"
    assert stem("users") == "user"
    assert stem("create") == "creat"
    assert stem("created") == "creat"


def test_query_terms_are_distinct_and_keep_first_occurrence_order():
    assert query_terms("validate the token, validating tokens") == ["valid", "token"]


def test_bm25_weights_name_over_doc_over_file_and_reports_matched_fields():
    index = CodeSearchIndex(
        [
            ("in-name", {"name": "auth.validate_token", "doc": None, "file": "auth/a.py"}),
            ("in-doc", {"name": "auth.check", "doc": "Validates a token.", "file": "auth/b.py"}),
            ("in-file", {"name": "auth.run", "doc": None, "file": "auth/token_validation.py"}),
            ("unrelated", {"name": "orders.place", "doc": "Places an order.", "file": "o.py"}),
        ]
    )

    matches = index.search(query_terms("validate token"))

    assert set(matches) == {"in-name", "in-doc", "in-file"}
    ranked = sorted(matches, key=lambda key: -matches[key].score)
    assert ranked == ["in-name", "in-doc", "in-file"]
    assert matches["in-name"].fields == ("name",)
    assert matches["in-doc"].fields == ("doc",)
    assert matches["in-file"].fields == ("file",)
    assert index.search(["missing"]) == {}


def test_bm25_idf_and_field_length_normalisation():
    index = CodeSearchIndex(
        [
            ("short", {"name": "token"}),
            ("long", {"name": "token cache entry builder factory"}),
            ("other", {"name": "order"}),
        ]
    )

    # Two of three documents hold "token": ln(1 + (3 - 2 + 0.5) / (2 + 0.5)).
    assert index.idf("token") == pytest.approx(math.log(1 + 1.5 / 2.5))
    matches = index.search(["token"])
    # Same term frequency, shorter name field: the shorter document wins.
    assert matches["short"].score > matches["long"].score
    assert len(index) == 3


def test_summary_field_is_searched_for_flows():
    index = CodeSearchIndex(
        [
            ("flow", {"name": "POST /orders", "summary": "Checks the token, then charges."}),
            ("symbol", {"name": "orders.place"}),
        ]
    )

    assert index.search(query_terms("charge"))["flow"].fields == ("summary",)
