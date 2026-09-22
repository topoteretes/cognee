from datetime import datetime, timezone

from cognee.tasks.memify.cognify_session import session_memory_filename

SYNCED_AT = datetime(2026, 9, 22, 10, 14, 33, tzinfo=timezone.utc)


def test_filename_carries_timestamp_tag_and_session_id():
    name = session_memory_filename("claude_8a46f4fc-3450-4b9b-859b-b277d203ea45", SYNCED_AT)
    assert (
        name
        == "2026-09-22T10-14-33Z_session-memory_claude_8a46f4fc-3450-4b9b-859b-b277d203ea45.txt"
    )


def test_timestamp_is_rendered_in_utc():
    from datetime import timedelta

    cest = timezone(timedelta(hours=2))
    local = datetime(2026, 9, 22, 12, 14, 33, tzinfo=cest)
    assert session_memory_filename("s", local).startswith("2026-09-22T10-14-33Z_")


def test_unsafe_characters_in_session_id_are_folded():
    name = session_memory_filename("default_session:abc/def ghi", SYNCED_AT)
    assert name.endswith("_session-memory_default_session-abc-def-ghi.txt")


def test_empty_session_id_still_yields_a_usable_name():
    assert session_memory_filename("///", SYNCED_AT).endswith("_session-memory_session.txt")


def test_defaults_to_now_when_no_time_is_given():
    before = datetime.now(timezone.utc).replace(microsecond=0)
    name = session_memory_filename("s")
    stamp = datetime.strptime(name.split("_", 1)[0], "%Y-%m-%dT%H-%M-%SZ").replace(
        tzinfo=timezone.utc
    )
    assert stamp >= before
