import json

import pytest

from cognee.eval_framework.benchmark_adapters.locomo_adapter import (
    ADVERSARIAL_GOLD_ANSWER,
    LocomoAdapter,
    build_question_records,
    category_name,
    format_turn,
    is_abstention,
    parse_conversation,
    parse_evidence,
)
from cognee.eval_framework.locomo.preprocess import (
    build_conversation_windows,
    build_session_windows,
    conversation_overview_text,
    dataset_name_for,
)


def make_record(sample_id="conv-1", turns_per_session=(4, 3)):
    conversation = {"speaker_a": "Ana", "speaker_b": "Ben"}
    for session_index, count in enumerate(turns_per_session, start=1):
        conversation[f"session_{session_index}_date_time"] = f"1:00 pm on {session_index} May, 2023"
        conversation[f"session_{session_index}"] = [
            {
                "speaker": "Ana" if i % 2 == 0 else "Ben",
                "dia_id": f"D{session_index}:{i + 1}",
                "text": f"turn {i + 1} of session {session_index}",
                **({"blip_caption": "a cat on a sofa", "img_url": "['x']"} if i == 1 else {}),
            }
            for i in range(count)
        ]
    qa = [
        {"question": "Q single?", "answer": "A single", "evidence": "['D1:1']", "category": "4"},
        {
            "question": "Q multi?",
            "answer": "A multi",
            "evidence": "['D1:2', 'D2:1']",
            "category": 1,
        },
        {"question": "Q when?", "answer": "1 May 2023", "evidence": "['D1:3']", "category": "2"},
        {
            "question": "Q adversarial?",
            "adversarial_answer": "distractor",
            "evidence": "['D2:3']",
            "category": "5",
        },
    ]
    return {"sample_id": sample_id, "conversation": conversation, "qa": qa}


def test_parse_evidence_accepts_stringified_lists_and_lists():
    assert parse_evidence("['D1:2', 'D2:1']") == ["D1:2", "D2:1"]
    assert parse_evidence(["D1:2"]) == ["D1:2"]
    assert parse_evidence("") == []
    assert parse_evidence(None) == []


def test_category_names():
    assert category_name("1") == "multi_hop"
    assert category_name(2) == "temporal"
    assert category_name(3) == "open_domain"
    assert category_name(4) == "single_hop"
    assert category_name(5) == "adversarial"
    assert category_name("x") == "category_x"


def test_parse_conversation_and_format_turn():
    conversation = parse_conversation(make_record(), 0)
    assert conversation.speaker_a == "Ana"
    assert [s.index for s in conversation.sessions] == [1, 2]
    assert conversation.sessions[0].date_time == "1:00 pm on 1 May, 2023"
    assert conversation.turn_count == 7
    photo_turn = conversation.sessions[0].turns[1]
    assert format_turn(photo_turn) == "Ben: turn 2 of session 1 [shares a photo: a cat on a sofa]"


def test_build_question_records_adversarial_and_golden_context():
    conversation = parse_conversation(make_record(), 3)
    records = build_question_records(conversation, load_golden_context=True)
    assert [r["question_type"] for r in records] == [
        "single_hop",
        "multi_hop",
        "temporal",
        "adversarial",
    ]
    assert all(r["conversation_index"] == 3 for r in records)
    assert records[0]["question_idx"] == 0
    assert "Session 1, 1:00 pm on 1 May, 2023" in records[0]["golden_context"]
    adversarial = records[-1]
    assert adversarial["answer"] == ADVERSARIAL_GOLD_ANSWER
    assert adversarial["adversarial_answer"] == "distractor"
    # Questions are asked as written: no augmented completion question, no answer options.
    assert all("completion_question" not in r and "answer_options" not in r for r in records)


def test_build_question_records_drops_beyond_truncation_and_adversarial():
    conversation = parse_conversation(make_record(), 0)
    records = build_question_records(conversation, include_adversarial=False, max_session_index=1)
    # multi-hop needs D2:1 -> dropped; adversarial dropped by flag
    assert [r["question_type"] for r in records] == ["single_hop", "temporal"]


def test_adapter_loads_conversations_and_questions_from_file(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps([make_record("conv-1"), make_record("conv-2")]), encoding="utf-8")

    adapter = LocomoAdapter(data_path=str(path))
    assert adapter.conversation_count() == 2
    conversation = adapter.load_conversation(1)
    assert conversation.sample_id == "conv-2"
    questions = adapter.questions_for(conversation)
    assert len(questions) == 4
    assert all(q["conversation_index"] == 1 for q in questions)

    # max_sessions truncates the conversation and drops questions that need a later session
    truncated = LocomoAdapter(max_sessions=1, data_path=str(path))
    conversation = truncated.load_conversation(1)
    assert [s.index for s in conversation.sessions] == [1]
    assert [q["question_type"] for q in truncated.questions_for(conversation)] == [
        "single_hop",
        "temporal",
    ]

    with pytest.raises(IndexError):
        truncated.load_conversation(5)


def test_session_windows_have_dated_headers_and_fold_small_tail():
    conversation = parse_conversation(make_record(turns_per_session=(7,)), 0)
    session = conversation.sessions[0]
    windows = build_session_windows(conversation, session, window_turns=3)
    # 7 turns / 3 -> [3, 3, 1]; the 1-turn tail folds into the previous window -> 2 windows
    assert [w.parts for w in windows] == [2, 2]
    assert windows[0].text.startswith(
        "Session 1 of the conversation between Ana and Ben, which took place at 1:00 pm on 1 May, 2023"
    )
    assert windows[1].last_dia_id == "D1:7"
    assert "Ben: turn 2 of session 1 [shares a photo: a cat on a sofa]" in windows[0].text
    assert build_session_windows(conversation, session, window_turns=100)[0].parts == 1
    with pytest.raises(ValueError):
        build_session_windows(conversation, session, window_turns=0)


def test_conversation_windows_and_overview():
    conversation = parse_conversation(make_record(), 0)
    windows = build_conversation_windows(conversation, window_turns=2)
    # session 1: 4 turns -> 2 windows; session 2: 3 turns -> [2, 1], the 1-turn tail folds in
    assert [(w.session_index, w.part, w.parts) for w in windows] == [
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 1),
    ]
    assert "(part 1 of 2)" in windows[0].text
    assert "(part" not in windows[2].text
    assert windows[2].last_dia_id == "D2:3"
    assert dataset_name_for(conversation) == "locomo_conv_1"
    assert "Session 2 took place at 1:00 pm on 2 May, 2023" in conversation_overview_text(
        conversation
    )


def test_is_abstention():
    assert is_abstention("Not mentioned")
    assert is_abstention("No information available in the chat")
    assert not is_abstention("a necklace")
    assert not is_abstention(None)
