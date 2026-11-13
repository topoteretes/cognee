"""Set up synthetic sample financial documents for the TrustRAG cookbook.

Writes three synthetic financial documents to sample/ (git-ignored):
- Regulatory-Notice-24-99-synthetic.txt: A synthetic FINRA regulatory notice on algorithmic risk controls.
- Master-Services-Agreement-synthetic.txt: A synthetic CUAD commercial contract excerpt.
- Acme-Financial-10K-synthetic.txt: A synthetic EDGAR Form 10-K annual report excerpt.

All documents are entirely synthetic, ensuring 100% license compliance and redistributability.
No real EDGAR, FINRA, or CUAD confidential data is required.

Run: uv run python examples/cookbooks/trustrag/setup.py
"""

from __future__ import annotations

import os
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample"

SYNTHETIC_FINRA_NOTICE = """FINRA Regulatory Notice 24-99
Financial Industry Regulatory Authority
Published: October 15, 2024
Topic: Supervisory Controls for Automated and Algorithmic Order Execution Systems

Executive Summary:
FINRA is issuing Regulatory Notice 24-99 to remind member firms of their supervisory obligations under FINRA Rule 3110 regarding algorithmic trading and automated execution systems. As electronic trading environments evolve, member firms must establish, maintain, and enforce written supervisory procedures reasonably designed to supervise algorithmic activities.

Key Operational Requirements:
1. Annual Stress Testing: Member firms deploying algorithmic execution models must conduct mandatory stress testing at least once every calendar year. The testing suite must simulate extreme market volatility and include a minimum simulated market liquidity shock parameter of 35%.
2. Mandatory Pre-Trade Risk Checks: Execution systems must enforce real-time, automated pre-trade risk controls. Automated controls must immediately reject any individual single order with a notional value exceeding $10,000,000, unless the order receives explicit real-time approval from a designated General Securities Principal.
3. Automated Kill-Switch Functionality: Algorithms must incorporate an independent automated kill-switch capable of terminating execution within 50 milliseconds of detecting erroneous order patterns or anomalous market feeds.
4. Record Retention Schedule: In accordance with SEA Rule 17a-4 and FINRA Rule 4511, all algorithmic design specifications, validation test records, model changes, and real-time execution logs must be retained for an immutable retention period of not less than seven years.
"""

SYNTHETIC_CUAD_CONTRACT = """MASTER SERVICES AGREEMENT
This Master Services Agreement ("Agreement") is entered into and made effective as of January 1, 2025 ("Effective Date"), by and between Meridian Financial Solutions Inc., a Delaware corporation ("Customer"), and Apex Cloud Technologies LLC, a Delaware limited liability company ("Service Provider").

RECITALS
WHEREAS, Service Provider provides high-throughput financial data management and analytics infrastructure; and
WHEREAS, Customer desires to retain Service Provider to perform such cloud infrastructure services in accordance with the terms herein.

TERMS AND CONDITIONS
1. Term and Renewal: The initial term of this Agreement shall commence on the Effective Date and continue in full force for a period of thirty-six (36) consecutive months ("Initial Term"). Thereafter, this Agreement shall automatically renew for successive renewal terms of twelve (12) months each, unless either party provides written notice of non-renewal at least ninety (90) days prior to the expiration of the then-current term.
2. Termination for Convenience: Customer may terminate this Agreement or any Statement of Work for convenience and without cause upon providing at least ninety (90) days prior written notice to Service Provider.
3. Limitation of Liability: Under Section 11.2, except for gross negligence, willful misconduct, or breach of confidentiality obligations under Section 8, the aggregate liability of Service Provider arising out of or related to this Agreement shall be strictly capped at $5,000,000 or the total fees paid by Customer to Service Provider in the preceding twelve (12) months, whichever amount is greater.
4. Mutual Indemnification: Each party agrees to defend, indemnify, and hold harmless the other party and its officers, directors, and employees from and against any third-party claims, demands, and liabilities alleging that the indemnifying party's software or materials infringe any registered patent, copyright, or trademark, subject to an aggregate indemnification liability ceiling of $10,000,000.
"""

SYNTHETIC_EDGAR_10K = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the Fiscal Year Ended December 31, 2024
Commission File Number: 001-98765

ACME FINANCIAL TECHNOLOGIES INC.
State of Incorporation: Delaware | IRS Employer Identification No.: 12-3456789
Principal Executive Offices: 100 Wall Street, 28th Floor, New York, NY 10005

PART I - ITEM 1. BUSINESS OVERVIEW
Acme Financial Technologies Inc. is a leading provider of real-time enterprise risk management software and regulatory reporting infrastructure for global capital markets institutions.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
Fiscal Year 2024 Financial Highlights:
- Revenue: Total consolidated revenue for the fiscal year ended December 31, 2024 was $450,000,000, representing an increase of 18.5% compared to total consolidated revenue of $380,000,000 for the fiscal year ended December 31, 2023.
- Net Income: Consolidated net income for fiscal year 2024 was $62,500,000, representing an increase of 29.7% compared to net income of $48,200,000 reported for fiscal year 2023.
- Operating Margin: Operating margin expanded by 240 basis points to 17.8% for fiscal year 2024, driven by cloud infrastructure efficiencies and subscription licensing scale.
- Cash and Liquidity: As of December 31, 2024, the Company maintained cash, cash equivalents, and marketable securities totaling $128,400,000, with zero outstanding long-term debt.
- Headcount: As of December 31, 2024, Acme Financial Technologies employed 1,420 full-time employees across its global development and client support centers.
"""


def write_sample(target_dir: Path | None = None) -> list[Path]:
    """Write synthetic financial documents to sample directory."""
    out_dir = target_dir or SAMPLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (out_dir / "Regulatory-Notice-24-99-synthetic.txt", SYNTHETIC_FINRA_NOTICE),
        (out_dir / "Master-Services-Agreement-synthetic.txt", SYNTHETIC_CUAD_CONTRACT),
        (out_dir / "Acme-Financial-10K-synthetic.txt", SYNTHETIC_EDGAR_10K),
    ]

    written_paths: list[Path] = []
    for path, content in files:
        path.write_text(content.strip(), encoding="utf-8")
        written_paths.append(path)
        print(f"[setup] Wrote synthetic document: {path.name} ({len(content)} chars)")

    return written_paths


if __name__ == "__main__":
    print(f"[setup] Setting up synthetic financial sample in {SAMPLE_DIR}...")
    written = write_sample()
    print(f"[setup] Successfully created {len(written)} synthetic financial documents.")

