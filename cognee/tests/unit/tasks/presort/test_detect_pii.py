from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.tasks.presort.detect_pii import (
    PiiAssessment,
    PiiCategoryAssessment,
    detect_pii,
    iban_valid,
    luhn_valid,
    redact,
    scan_content,
    scan_filename,
)
from cognee.tasks.presort.models import FileRecord


def _text_record(tmp_path, name: str, text: str) -> FileRecord:
    path = tmp_path / name
    path.write_text(text)
    return FileRecord(
        path=str(path), name=name, extension=name.rsplit(".", 1)[-1], is_text=True, size_bytes=1
    )


def test_luhn():
    assert luhn_valid("4111111111111111")  # classic Visa test number
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("1234")  # too short


def test_iban():
    assert iban_valid("DE89370400440532013000")
    assert not iban_valid("DE89370400440532013001")


def test_redact_email_and_generic():
    assert redact("jane.doe@example.com") == "j***@example.com"
    assert redact("4111111111111111") == "41***11"
    assert redact("ab") == "***"


def test_filename_hints():
    record = FileRecord(path="/x/Passport-Scan (1).pdf", name="Passport-Scan (1).pdf")
    findings = scan_filename(record)
    assert any(f.category == "identity_document" and f.severity == "high" for f in findings)

    boring = FileRecord(path="/x/holiday.jpg", name="holiday.jpg")
    assert scan_filename(boring) == []


def test_content_regexes():
    record = FileRecord(path="/x/notes.txt", name="notes.txt", is_text=True)
    text = (
        "Reach me at jane.doe@example.com or +1 555 123 4567.\n"
        "IBAN: DE89370400440532013000. SSN 123-45-6789.\n"
        "Card: 4111 1111 1111 1111. Token: sk-abcdefabcdefabcdef123456\n"
        "Invalid card: 4111 1111 1111 1112."
    )
    findings = scan_content(record, text)
    categories = {finding.category for finding in findings}
    assert categories == {
        "email_address",
        "phone_number",
        "bank_account",
        "government_id_number",
        "payment_card",
        "credential",
    }
    for finding in findings:
        if finding.redacted_sample:
            assert "jane.doe@example.com" not in finding.redacted_sample
            assert "4111 1111 1111 1111" not in finding.redacted_sample


@pytest.mark.asyncio
async def test_detect_pii_deterministic(tmp_path):
    record = _text_record(tmp_path, "notes.txt", "email me: ada@example.com")
    clean = _text_record(tmp_path, "clean.txt", "nothing personal here")

    findings = await detect_pii([record, clean])

    assert {finding.path for finding in findings} == {record.path}
    assert findings[0].category == "email_address"


@pytest.mark.asyncio
async def test_detect_pii_llm_layer(tmp_path):
    record = _text_record(tmp_path, "medical_notes.txt", "patient history follows")

    assessment = PiiAssessment(
        contains_personal_data=True,
        categories=[
            PiiCategoryAssessment(
                category="medical_document", severity="high", rationale="contains a diagnosis"
            )
        ],
    )
    with patch.object(
        LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=assessment),
    ) as llm_mock:
        findings = await detect_pii([record], use_llm=True)

    llm_mock.assert_awaited_once()
    llm_findings = [finding for finding in findings if finding.source == "llm"]
    assert len(llm_findings) == 1
    assert llm_findings[0].category == "medical_document"


@pytest.mark.asyncio
async def test_llm_not_called_without_flag_or_findings(tmp_path):
    clean = _text_record(tmp_path, "clean.txt", "nothing personal here")
    with patch.object(LLMGateway, "acreate_structured_output", new=AsyncMock()) as llm_mock:
        await detect_pii([clean], use_llm=True)  # no deterministic findings -> no LLM call
        await detect_pii([clean], use_llm=False)
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_failure_is_swallowed(tmp_path):
    record = _text_record(tmp_path, "cv.txt", "resume of Ada, ada@example.com")
    with patch.object(
        LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=RuntimeError("no llm")),
    ):
        findings = await detect_pii([record], use_llm=True)
    # Deterministic findings survive; the failure lands in record warnings.
    assert any(finding.source in ("filename", "content") for finding in findings)
    assert any("LLM PII assessment failed" in warning for warning in record.warnings)
