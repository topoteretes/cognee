"""Financial and regulatory knowledge graph model for Cognee.

Defines domain-specific node types for financial intelligence:
- Organization (corporations, regulatory agencies, financial institutions)
- Person (officers, directors, regulators, signatories)
- RegulatoryFiling (10-K, 10-Q, 8-K, FINRA regulatory notices, CUAD contracts)
- Regulation (statutes, rules, regulatory mandates)
- FinancialInstrument (equities, debt, credit facilities, derivatives)

Each node declares identity_fields so entities with identical identities
automatically merge into single nodes across diverse document sources.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import field_validator

from cognee.low_level import DataPoint, FromIdentity


def _clean_name(name: str) -> str:
    """Normalize names by stripping common extraneous prefixes."""
    name = re.sub(r"^(the|an|a)\s+", "", name.strip(), flags=re.IGNORECASE)
    return name.strip()


class Organization(DataPoint):
    name: str
    jurisdiction: str | None = None
    industry: str | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, name: str) -> str:
        return _clean_name(name)


class Person(DataPoint):
    name: str
    role: str | None = None
    associated_organization: Annotated[Organization, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    @field_validator("name")
    @classmethod
    def _normalize_person(cls, name: str) -> str:
        return _clean_name(name)


class RegulatoryFiling(DataPoint):
    filing_id: str
    title: str | None = None
    filing_type: str | None = None  # e.g., 10-K, 10-Q, 8-K, Regulatory Notice, Agreement
    filed_by: Annotated[Organization, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["filing_id", "title"], "identity_fields": ["filing_id"]}


class Regulation(DataPoint):
    name: str
    summary: str | None = None
    enforced_by: Annotated[Organization, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class FinancialInstrument(DataPoint):
    name: str
    instrument_type: str | None = None  # e.g., Equity, Debt, Credit Facility, Derivative
    issued_by: Annotated[Organization, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class FinancialGraph(DataPoint):
    """Top-level container for all extracted financial and regulatory entities."""

    organizations: list[Organization] = []
    people: list[Person] = []
    filings: list[RegulatoryFiling] = []
    regulations: list[Regulation] = []
    instruments: list[FinancialInstrument] = []


EXTRACTION_PROMPT = """
Extract organizations, key persons, regulatory filings, rules/regulations, and financial instruments
mentioned in the text.

1. Organizations: Companies, regulatory authorities (e.g. SEC, FINRA), issuers, and counterparties.
2. People: Officers, executives, signatories, and key individuals with their professional role.
3. Regulatory Filings: Filings or notices identified by number/title (e.g., '10-K', 'Form 8-K', 'Notice 18-03', or contract titles).
4. Regulations: Explicit laws, acts, or rules referenced (e.g., 'Securities Exchange Act', 'FINRA Rule 2210').
5. Financial Instruments: Securities, debt obligations, credit agreements, or financial facilities described.

Use exact names without unnecessary articles ('the'). Place all extracted entities and those referenced into the matching top-level lists.
"""

