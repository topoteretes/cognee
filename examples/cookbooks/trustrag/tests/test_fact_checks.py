"""Unit tests for deterministic fact checking."""

from __future__ import annotations

import pytest

from examples.cookbooks.trustrag.core.fact_checks import (
    FactCheckStatus,
    FactCheckType,
    check_claim_against_chunks,
    extract_currencies,
    extract_dates,
    extract_numbers,
    extract_percentages,
)


def test_currency_check_match():
    claim = "The company incurred a fine of $150,000 for compliance violations."
    chunk = "FINRA ordered Member Firm to pay a fine of $150,000 and submit to censure."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is True
    curr_checks = [c for c in res.checks if c.check_type == FactCheckType.CURRENCY]
    assert len(curr_checks) == 1
    assert curr_checks[0].status == FactCheckStatus.PASSED


def test_currency_check_mismatch():
    claim = "The settlement amount was $500,000."
    chunk = "Respondent agreed to pay a monetary penalty of $250,000."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is False
    assert "Currency mismatch" in (res.downgrade_reason or "")


def test_percentage_check_match():
    claim = "Operating margins increased by 14.5% year over year."
    chunk = "For the fiscal year, operating margin expanded by 14.5% compared to prior year."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is True
    pct_checks = [c for c in res.checks if c.check_type == FactCheckType.PERCENTAGE]
    assert len(pct_checks) == 1
    assert pct_checks[0].status == FactCheckStatus.PASSED


def test_percentage_check_mismatch():
    claim = "Revenues grew by 25%."
    chunk = "Total revenues grew by 12% in the third quarter."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is False


def test_date_check_match():
    claim = "The agreement was executed on January 15, 2024."
    chunk = "Dated as of January 15, 2024, by and between Company and Buyer."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is True


def test_date_check_mismatch():
    claim = "The filing date was October 12, 2023."
    chunk = "Filed with the SEC on November 15, 2023."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is False


def test_number_check_match():
    claim = "The board consists of 9 independent directors."
    chunk = "As of December 31, the board comprised 9 independent directors and the CEO."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is True


def test_number_check_mismatch():
    claim = "The firm has 15 branch offices."
    chunk = "The firm currently operates 22 branch offices across the Midwest."
    res = check_claim_against_chunks(claim, [chunk])
    assert res.passed is False

