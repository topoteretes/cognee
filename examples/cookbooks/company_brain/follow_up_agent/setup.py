"""Set up sample calls, Linear issues and email, so the cookbook runs before you connect yours.

Writes sample/ in the shape Granola's API, Linear's API and the Gmail connector give the
scripts. Dates are relative to today, so the latest call is always two days ago. Only
LLM_API_KEY is needed; no Granola, Linear, Gmail or Slack account. A sample run never posts
to Slack.

The story: in today's latest call, "Checkout v2 launch readiness", Omar agrees to move card
payments to 3DS2 and Sam to load-test the checkout API. The call says neither team nor
deadline. An earlier call says Omar is on the Payments team, Linear already tracks his step
as PAY-104, and an email from the acquiring bank sets the 3DS2 deadline.

Run: uv run python examples/cookbooks/company_brain/follow_up_agent/setup.py
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"


def day(days_ago: int) -> str:
    return (datetime.now().astimezone().date() - timedelta(days=days_ago)).isoformat()


SAMPLE_FILES = {
    "calls/01_payments_weekly.txt": f"""Call: Payments weekly
Date: {day(20)}
Attendees: Lena Fischer, Omar Haddad, Sam Okoro

Lena Fischer leads the Payments team. Omar Haddad, on the Payments team, owns card
processing. Sam Okoro joined from the Platform team to talk about checkout capacity.

Lena Fischer: Omar, card processing stays with you through the checkout v2 launch.
Sam Okoro: Platform can run the load tests once the API is frozen.""",
    "calls/02_checkout_v2_launch_readiness.txt": f"""Call: Checkout v2 launch readiness
Date: {day(2)}
Attendees: Lena Fischer, Omar Haddad, Sam Okoro

The team reviewed what has to happen before checkout v2 goes live.

Lena Fischer: What's blocking launch?
Omar Haddad: Card payments still use the old 3-D Secure flow. I'll migrate them to 3DS2.
Sam Okoro: I'll load-test the checkout API at three times peak traffic.
Lena Fischer: And I'll write the launch announcement once both are done.""",
    "linear/PAY-104.txt": f"""Linear issue PAY-104: Migrate card payments to 3DS2
Status: In Progress
Team: Payments
Project: Checkout v2
Assignee: Omar Haddad
Due: none

Card payments still go through the 3-D Secure 1 flow. Move them to 3DS2. Updated {day(5)}.""",
    "linear/PLAT-88.txt": f"""Linear issue PLAT-88: Load-test the checkout API
Status: Todo
Team: Platform
Project: Checkout v2
Assignee: Sam Okoro
Due: {day(-10)}

Run a load test against the checkout API at three times peak traffic.""",
    "email/01_kestrel_bank_3ds2.txt": f"""Subject: 3DS2 required for card payments
From: Kestrel Bank Merchant Services <merchants@kestrel.example>
To: Lena Fischer <lena@acorn.example>
Date: {day(7)}

Dear merchant,

From {day(-30)}, every card payment must use 3DS2. Payments that still use 3-D Secure 1
after that date will be declined.

Kestrel Bank Merchant Services""",
}


def write_sample() -> None:
    """Write every sample file, replacing an earlier sample/ folder."""
    shutil.rmtree(SAMPLE, ignore_errors=True)
    for name, text in SAMPLE_FILES.items():
        path = SAMPLE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


if __name__ == "__main__":
    write_sample()
    print(f"[setup] Wrote {len(SAMPLE_FILES)} sample files to sample/.")
    print(
        "[setup] Now run: uv run python "
        "examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --sample"
    )
