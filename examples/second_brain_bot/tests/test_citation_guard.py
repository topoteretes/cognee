"""Deterministic pressure-test for the citation guard (no cognee, no keys).

This is the regression net for the bug that resurfaced twice: a "no information"
answer receiving a citation. cognee's answer-grounded Evidence block can name a
note's data_id even when the graph refused to answer (the note is topically
near), so citing purely from Evidence is not enough. select_citations adds a
structural gate: the answer must use a distinctive term from the note (one not
already in the query). A refusal only echoes the query subject, so it cites
nothing, no matter how it is phrased. These tests lock that in across phrasings.
"""

import uuid

from second_brain_bot.adapter.cognee_adapter import _CitationRecord, select_citations

_NOTE = "The Meridian archive opens on September 5th in Salem."
_QUERY = "When does the Meridian archive open?"
_DATA_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "meridian")

# An Evidence block that DOES name the note (the hard case: retrieval matched).
_EVIDENCE = f'- chunk 1 of document text_abc (data_id: {_DATA_ID}, chunk_id: def): "{_NOTE}"'


def _record():
    return _CitationRecord(
        text=_NOTE,
        transport="telegram",
        source="chat88",
        ts="2026-06-12T08:30:00",
        deeplink="telegram://chat88/903",
        data_id=_DATA_ID,
    )


def test_real_answer_is_cited_to_source():
    cites = select_citations([_record()], _EVIDENCE, "September 5th.", _QUERY)
    assert len(cites) == 1
    assert cites[0].source_transport == "telegram"
    assert cites[0].source_ref == "telegram://chat88/903"


# Every one of these is a refusal that ECHOES the query subject and must cite
# nothing, even though the note's data_id is present in the Evidence block.
_REFUSALS = [
    "There is no information about the Meridian archive in the provided context.",
    "The context does not mention the Meridian archive.",
    "Based on the context, I cannot determine when the Meridian archive opens.",
    "The provided context does not contain details about the Meridian archive.",
    "I don't have information on the Meridian archive.",
    "No relevant information about the Meridian archive was found.",
    "Sorry, the Meridian archive is not covered by the available context.",
]


def test_refusals_are_never_cited_regardless_of_phrasing():
    record = _record()
    for refusal in _REFUSALS:
        cites = select_citations([record], _EVIDENCE, refusal, _QUERY)
        assert cites == [], f"refusal was cited: {refusal!r}"


def test_no_evidence_means_no_citation():
    # Even a perfectly on-topic answer cites nothing if cognee retrieved nothing.
    cites = select_citations([_record()], "", "September 5th.", _QUERY)
    assert cites == []


def test_note_not_in_evidence_is_not_cited():
    # data_id absent from Evidence and text not quoted -> not retrieved -> no cite.
    other_evidence = '- chunk 1 of document text_zzz (data_id: 00000000-0000-0000-0000-000000000000): "unrelated"'
    cites = select_citations([_record()], other_evidence, "September 5th.", _QUERY)
    assert cites == []


# The distinctive gate bets a refusal only echoes the query subject. That fails
# when refusal boilerplate ("found", "available", ...) happens to be a content
# word of the note too, so those surface forms are stopworded. This is the case
# the earlier phrasing net missed because its fixture note had no such word.
_BOILERPLATE_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "passport")
_BOILERPLATE_NOTE = "The missing passport was found in the drawer."
_BOILERPLATE_EVIDENCE = (
    f'- chunk 1 of document text_p (data_id: {_BOILERPLATE_ID}): "{_BOILERPLATE_NOTE}"'
)


def _boilerplate_record():
    return _CitationRecord(
        text=_BOILERPLATE_NOTE,
        transport="web",
        source="sess1",
        ts="2026-06-12T08:30:00",
        deeplink="web://sess1",
        data_id=_BOILERPLATE_ID,
    )


def test_refusal_sharing_boilerplate_word_with_note_is_not_cited():
    query = "Where is my passport?"
    for refusal in (
        "No relevant information about the passport was found.",
        "Sorry, the passport is not covered by the available context.",
    ):
        cites = select_citations([_boilerplate_record()], _BOILERPLATE_EVIDENCE, refusal, query)
        assert cites == [], f"refusal was cited: {refusal!r}"


def test_real_answer_still_cited_when_note_contains_boilerplate_word():
    # The note has boilerplate words, but a genuine answer shares a distinctive
    # content term ("drawer"), so it is still cited.
    cites = select_citations(
        [_boilerplate_record()], _BOILERPLATE_EVIDENCE, "In the drawer.", "Where is my passport?"
    )
    assert len(cites) == 1
    assert cites[0].source_transport == "web"
