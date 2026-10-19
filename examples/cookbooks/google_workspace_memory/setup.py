"""Set up a sample Google Workspace, so the cookbook runs before you connect your own.

Writes sample/: Ana Novak's calendar events, inbox emails and her orders sheet from Drive,
in the shape the Google Calendar, Gmail and Google Drive connectors give the scripts. Dates
are relative to today, so the client meetings are always in the coming days. Only
LLM_API_KEY is needed; no Google account.

The story: Ana runs Birchwood Ceramics, which makes tableware for restaurants. She meets
Harbor Café in two days and Olive & Fig in four. Harbor Café's dinner plates were promised
for tomorrow, but the courier delayed them by two days, the day of their menu launch. Olive &
Fig's pasta bowls need a sage glaze that the supplier back-ordered past the promised date.

Run: uv run python examples/cookbooks/google_workspace_memory/setup.py
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"
ANA = "Ana Novak <ana@birchwood.example>"
LEO = "Leo Marsh <leo@harborcafe.example>"
SOFIA = "Sofia Reyes <sofia@oliveandfig.example>"


def day(offset: int) -> str:
    """A date relative to today: negative in the past, positive in the future."""
    return (datetime.now().astimezone().date() + timedelta(days=offset)).isoformat()


def at(offset: int, hour: int, minute: int = 0) -> str:
    """A time in the format the Calendar connector writes."""
    return f"{day(offset)}T{hour:02d}:{minute:02d}:00+02:00 (Europe/Berlin)"


SAMPLE_FILES = {
    "drive/Orders.csv": f"""order_id,client,item,quantity,ordered,promised_delivery,status
BC-1041,Harbor Café,Espresso cups (stone grey),120,{day(-20)},{day(-3)},Delivered
BC-1042,Harbor Café,Dinner plates 27 cm (white),80,{day(-14)},{day(1)},Shipped
BC-1043,Olive & Fig,Pasta bowls (sage glaze),60,{day(-12)},{day(4)},In production
BC-1044,Olive & Fig,Serving platters (white),15,{day(-10)},{day(4)},In production
BC-1045,Northbank Hotel,Breakfast mugs (cobalt glaze),200,{day(-8)},{day(6)},Waiting for glaze
BC-1046,Harbor Café,Espresso saucers (stone grey),120,{day(-5)},{day(9)},Ordered""",
    "calendar/01_harbor_menu_planning.txt": f"""Calendar event: Spring menu planning with Harbor Café
Calendar: {ANA}
When: {at(-7, 10)} to {at(-7, 11)}
Location: Harbor Café, 12 Quay Street
Organizer: {ANA}
Attendees: {ANA} (accepted), {LEO} (accepted)

Leo launches the spring menu on {day(3)} and needs the new dinner plates (BC-1042) in the
restaurant before then. Ana confirmed delivery by {day(1)}.""",
    "calendar/02_harbor_quarterly_review.txt": f"""Calendar event: Quarterly review with Harbor Café
Calendar: {ANA}
When: {at(2, 10)} to {at(2, 11)}
Location: Harbor Café, 12 Quay Street
Organizer: {ANA}
Attendees: {ANA} (accepted), {LEO} (accepted)

Review open orders and next quarter's espresso line.""",
    "calendar/03_olive_fig_check_in.txt": f"""Calendar event: Order check-in with Olive & Fig
Calendar: {ANA}
When: {at(4, 14)} to {at(4, 14, 30)}
Organizer: {SOFIA}
Attendees: {ANA} (accepted), {SOFIA} (accepted)
Meeting link: https://meet.google.com/abc-defg-hij

Sofia wants to confirm the pasta bowls and platters arrive together for the new terrace.""",
    "calendar/04_kiln_schedule.txt": f"""Calendar event: Kiln schedule
Calendar: {ANA}
When: {at(-28, 8)} to {at(-28, 9)}
Organizer: {ANA}
Attendees: {ANA} (accepted), Tom Weber <tom@birchwood.example> (accepted)
Repeats: RRULE:FREQ=WEEKLY;BYDAY=MO

Plan the week's firings.""",
    "gmail/01_newsletter.txt": f"""Subject: This month in ceramics
From: Studio Pottery Monthly <news@studiopottery.example>
To: {ANA}
Date: {day(-3)}

Five glazes that are trending in restaurant tableware this season.""",
    "gmail/02_glaze_backorder.txt": f"""Subject: Sage glaze back-ordered
From: Clayworks Supply <orders@clayworks.example>
To: {ANA}
Date: {day(-2)}

Hi Ana,

The sage glaze on your order is back-ordered. We can deliver it on {day(5)}. The cobalt glaze
shipped today and arrives on {day(1)}.

Clayworks Supply""",
    "gmail/03_shipment_delayed.txt": f"""Subject: Shipment BC-1042 delayed
From: Swift Freight <tracking@swiftfreight.example>
To: {ANA}
Date: {day(-1)}

Your shipment BC-1042 (2 pallets) is held at the Riverside depot after a pallet was damaged
in loading. It will be repacked and delivered on {day(3)}.""",
    "gmail/04_plates_question.txt": f"""Subject: Plates before the menu launch?
From: {LEO}
To: {ANA}
Date: {day(-1)}

Hi Ana,

Just checking the dinner plates will be here before we launch the spring menu on {day(3)}.
We need them the evening before to set up.

Thanks,
Leo""",
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
        "examples/cookbooks/google_workspace_memory/google_workspace_memory.py --sample"
    )
