"""
Personal-data (PII) detection for presort.

Three layers, cheapest first:
- filename keywords ("passport", "payslip", ...) — no file reads;
- content regexes over a bounded text sample (emails, phones, IBAN with
  mod-97, SSN-like, credit cards with Luhn, API keys/tokens — the latter
  adapted from ``cognee/modules/operations/scrub_error.py``);
- an opt-in LLM pass (``use_llm=True``) that reviews flagged text files with a
  structured-output call for categories regexes cannot express.

Findings are inherently "potential personal data": regexes have false
positives and negatives. Samples stored in findings are always redacted;
raw matched text never enters the report.
"""

import asyncio
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.logging_utils import get_logger

from .models import FileRecord, PiiFinding

logger = get_logger("presort")

DEFAULT_MAX_SAMPLE_BYTES = 65536
_LLM_CONCURRENCY = 8

# filename keyword -> (category, severity)
FILENAME_HINTS: dict = {
    "passport": ("identity_document", "high"),
    "visa": ("identity_document", "high"),
    "id_card": ("identity_document", "high"),
    "idcard": ("identity_document", "high"),
    "driver_license": ("identity_document", "high"),
    "drivers_license": ("identity_document", "high"),
    "birth_certificate": ("identity_document", "high"),
    "ssn": ("government_id_number", "high"),
    "tax": ("financial_document", "high"),
    "payslip": ("financial_document", "high"),
    "salary": ("financial_document", "high"),
    "invoice": ("financial_document", "medium"),
    "bank_statement": ("financial_document", "high"),
    "bank": ("financial_document", "medium"),
    "iban": ("financial_document", "high"),
    "insurance": ("insurance_document", "medium"),
    "medical": ("medical_document", "high"),
    "prescription": ("medical_document", "high"),
    "diagnosis": ("medical_document", "high"),
    "cv": ("resume", "medium"),
    "resume": ("resume", "medium"),
    "lebenslauf": ("resume", "medium"),
    "contract": ("legal_document", "medium"),
    "agreement": ("legal_document", "medium"),
    "passport_photo": ("identity_document", "high"),
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+\d{1,3}[\s./-]?\(?\d{1,4}\)?(?:[\s./-]?\d{2,4}){2,4}")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# Credential shapes, adapted from scrub_error._SCRUB_PATTERNS.
_SECRET_RE = re.compile(
    r"\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9]{16,}\b|\bBearer\s+[A-Za-z0-9._-]{16,}\b",
    re.IGNORECASE,
)


def redact(sample: str, keep: int = 2) -> str:
    """Redact a matched value, keeping only a hint of its shape."""
    sample = sample.strip()
    if "@" in sample:  # email: keep first char and domain
        local, _, domain = sample.partition("@")
        return f"{local[:1]}***@{domain}"
    if len(sample) <= keep * 2:
        return "***"
    return f"{sample[:keep]}***{sample[-keep:]}"


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    rearranged = candidate[4:] + candidate[:4]
    digits = "".join(str(int(char, 36)) for char in rearranged)
    return int(digits) % 97 == 1


def scan_filename(record: FileRecord) -> List[PiiFinding]:
    findings = []
    normalized = re.sub(r"[\s()-]+", "_", record.name.lower())
    for keyword, (category, severity) in FILENAME_HINTS.items():
        if keyword in normalized:
            findings.append(
                PiiFinding(
                    path=record.path,
                    category=category,
                    severity=severity,
                    source="filename",
                    detail=f"filename contains {keyword!r}",
                )
            )
    return findings


def scan_content(record: FileRecord, text: str) -> List[PiiFinding]:
    findings = []

    def add(category: str, severity, sample: str, detail: str):
        findings.append(
            PiiFinding(
                path=record.path,
                category=category,
                severity=severity,
                source="content",
                redacted_sample=redact(sample),
                detail=detail,
            )
        )

    emails = _EMAIL_RE.findall(text)
    if emails:
        add("email_address", "low", emails[0], f"{len(emails)} email address(es)")

    phones = [match for match in _PHONE_RE.findall(text) if sum(c.isdigit() for c in match) >= 8]
    if phones:
        add("phone_number", "low", phones[0], f"{len(phones)} phone number(s)")

    ibans = [candidate for candidate in _IBAN_RE.findall(text) if iban_valid(candidate)]
    if ibans:
        add("bank_account", "high", ibans[0], f"{len(ibans)} IBAN(s)")

    ssns = _SSN_RE.findall(text)
    if ssns:
        add("government_id_number", "high", ssns[0], f"{len(ssns)} SSN-like number(s)")

    cards = [
        candidate
        for candidate in _CARD_RE.findall(text)
        if luhn_valid(re.sub(r"[ -]", "", candidate))
    ]
    if cards:
        add("payment_card", "high", cards[0], f"{len(cards)} card number(s) (Luhn-valid)")

    secrets = _SECRET_RE.findall(text)
    if secrets:
        add("credential", "high", secrets[0], f"{len(secrets)} API key/token shape(s)")

    return findings


class PiiCategoryAssessment(BaseModel):
    category: str = Field(description="Short snake_case category, e.g. medical_document")
    severity: Literal["low", "medium", "high"] = "medium"
    rationale: str = Field(description="One sentence; must not quote the personal data itself")


class PiiAssessment(BaseModel):
    contains_personal_data: bool
    categories: List[PiiCategoryAssessment] = Field(default_factory=list)


def _read_sample(record: FileRecord, max_sample_bytes: int) -> Optional[str]:
    try:
        with open(record.path, "rb") as file:
            return file.read(max_sample_bytes).decode("utf-8", errors="replace")
    except OSError as error:
        record.warnings.append(f"could not read sample for PII scan: {error}")
        return None


async def _llm_assess(record: FileRecord, text: str) -> List[PiiFinding]:
    system_prompt = render_prompt("detect_pii.txt", {"file_name": record.name})
    assessment = await LLMGateway.acreate_structured_output(text, system_prompt, PiiAssessment)
    if not assessment.contains_personal_data:
        return []
    return [
        PiiFinding(
            path=record.path,
            category=category.category,
            severity=category.severity,
            source="llm",
            detail=category.rationale,
        )
        for category in assessment.categories
    ]


async def detect_pii(
    files: List[FileRecord],
    *,
    use_llm: bool = False,
    max_sample_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
) -> List[PiiFinding]:
    findings: List[PiiFinding] = []
    flagged_text_records = []

    for record in files:
        record_findings = scan_filename(record)
        text = _read_sample(record, max_sample_bytes) if record.is_text else None
        if text:
            record_findings.extend(scan_content(record, text))
        findings.extend(record_findings)
        if use_llm and text and record_findings:
            flagged_text_records.append((record, text))

    if flagged_text_records:
        semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

        async def assess(record: FileRecord, text: str) -> List[PiiFinding]:
            async with semaphore:
                try:
                    return await _llm_assess(record, text)
                except Exception as error:  # LLM failures must not abort presort
                    record.warnings.append(f"LLM PII assessment failed: {error}")
                    logger.debug(f"Presort LLM PII failed for {record.path}: {error}")
                    return []

        results = await asyncio.gather(
            *(assess(record, text) for record, text in flagged_text_records)
        )
        for batch in results:
            findings.extend(batch)

    return findings
