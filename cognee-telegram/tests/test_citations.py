"""Citation ledger + t.me deep-link rendering (pure logic, no keys)."""

from cognee_telegram.citations import CitationLedger, MessageRef


def test_supergroup_deep_link_strips_100_prefix():
    ref = MessageRef(chat_id=-1001234567890, message_id=99, text="hi")
    assert ref.deep_link() == "https://t.me/c/1234567890/99"


def test_forum_topic_deep_link_includes_thread():
    ref = MessageRef(chat_id=-1001234567890, message_id=99, text="hi", thread_id=12)
    assert ref.deep_link() == "https://t.me/c/1234567890/12/99"


def test_dm_and_basic_group_have_no_public_link():
    assert MessageRef(chat_id=42, message_id=1, text="hi").deep_link() is None
    assert MessageRef(chat_id=-500, message_id=1, text="hi").deep_link() is None


def test_attributed_text_prefixes_author():
    assert (
        MessageRef(chat_id=1, message_id=1, text="hi", author="Ada").attributed_text() == "Ada: hi"
    )
    assert MessageRef(chat_id=1, message_id=1, text="hi").attributed_text() == "hi"


def test_ledger_records_and_resolves_by_overlap():
    ledger = CitationLedger()
    ds = "telegram_dm_7"
    ledger.record(
        ds, MessageRef(chat_id=7, message_id=1, text="the quarterly revenue report is due friday")
    )
    ledger.record(ds, MessageRef(chat_id=7, message_id=2, text="lunch plans for the weekend"))

    hits = ledger.resolve(ds, "when is the revenue report due")
    assert len(hits) == 1
    assert hits[0].message_id == 1


def test_ledger_abstains_when_nothing_overlaps():
    ledger = CitationLedger()
    ds = "telegram_dm_7"
    ledger.record(ds, MessageRef(chat_id=7, message_id=1, text="completely unrelated content"))
    assert ledger.resolve(ds, "xylophone zeppelin") == []


def test_ledger_drop_clears_dataset():
    ledger = CitationLedger()
    ds = "telegram_dm_7"
    ledger.record(ds, MessageRef(chat_id=7, message_id=1, text="something to remember"))
    ledger.drop(ds)
    assert ledger.refs(ds) == []
