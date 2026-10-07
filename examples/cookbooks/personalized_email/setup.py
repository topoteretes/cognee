"""Set up a sample mailbox and meetings, so the cookbook runs before you connect your own.

Writes sample/: Mira Lang's meeting notes, her sent mail and her inbox, in the shape the
Gmail connector and Granola's API give the scripts. Dates are relative to today, so the
newest email is always yesterday's. Only LLM_API_KEY is needed; no Gmail or Granola account.

The story: in a call twelve days ago, Mira promised Priya Shah at Northwind pilot pricing
by Friday and said SSO is SAML-only until OIDC ships next month. She never sent the
pricing. Yesterday Priya wrote asking about it.

Run: uv run python examples/cookbooks/personalized_email/setup.py
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"


def day(days_ago: int) -> str:
    return (datetime.now().astimezone().date() - timedelta(days=days_ago)).isoformat()


SAMPLE_FILES = {
    "meetings/01_northwind_pilot_scoping.txt": f"""Meeting: Northwind pilot scoping
Date: {day(12)}
Attendees: Mira Lang, Priya Shah, Jonas Berg

Northwind wants a pilot for three seats. Mira promised to send the pilot pricing by Friday:
3 seats at 400 EUR a month, first month free. SSO is SAML only for now; OIDC ships next
month. The pilot can start on the first of next month once the order form is signed.

Priya Shah: Our identity provider is moving to OIDC, so that matters to us.
Mira Lang: SAML works today; OIDC is next month. I'll send the pricing by Friday.""",
    "meetings/02_weekly_product_sync.txt": f"""Meeting: Weekly product sync
Date: {day(4)}
Attendees: Mira Lang, Jonas Berg

OIDC is on track and ships in the third week of next month. SAML works today with Okta and
Azure AD. Jonas can join a customer call to walk through the SSO setup.""",
    "sent_mail/01_to_jonas.txt": f"""Subject: Re: SSO timeline
From: Mira Lang <mira@lumen.example>
To: Jonas Berg <jonas@lumen.example>
Date: {day(9)}

Hi Jonas,

Thanks, that's clear. I'll tell Northwind next month for OIDC and not promise a date yet.

Best,
M.""",
    "sent_mail/02_to_tomas.txt": f"""Subject: Re: Renewal call
From: Mira Lang <mira@lumen.example>
To: Tomás Ruiz <tomas@harbor.example>
Date: {day(6)}

Hi Tomás,

Thursday at 10 works. I'll bring the usage numbers so we can size the renewal.

Best,
M.""",
    "sent_mail/03_to_priya.txt": f"""Subject: Great to meet you
From: Mira Lang <mira@lumen.example>
To: Priya Shah <priya@northwind.example>
Date: {day(12)}

Hi Priya,

Great talking today. I'll get the pilot pricing to you by Friday.

Best,
M.""",
    "inbox/01_newsletter.txt": f"""Subject: This week in SaaS pricing
From: Pricing Weekly <news@pricingweekly.example>
To: Mira Lang <mira@lumen.example>
Date: {day(3)}

Five ways to structure a free pilot without leaving money on the table.""",
    "inbox/02_pilot_start_and_sso.txt": f"""Subject: Pilot start and SSO
From: Priya Shah <priya@northwind.example>
To: Mira Lang <mira@lumen.example>
Date: {day(1)}

Hi Mira,

Two questions before we sign: can we start the pilot on the first of next month, and does
your SSO support OIDC? Our identity provider is OIDC-only from next quarter.

Also, I haven't seen the pilot pricing yet. Could you send it?

Thanks,
Priya""",
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
        "examples/cookbooks/personalized_email/personalized_email.py --sample"
    )
