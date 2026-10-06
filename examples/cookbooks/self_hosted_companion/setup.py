"""Set up a sample notes folder, so the companion runs before you point it at your own notes.

Writes sample/notes/: a few journal entries, dated relative to today, like the .md files in
an Obsidian vault or a journal folder. Only LLM_API_KEY is needed.

Run: uv run python examples/cookbooks/self_hosted_companion/setup.py
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"


def day(days_ago: int) -> str:
    return (datetime.now().astimezone().date() - timedelta(days=days_ago)).isoformat()


SAMPLE_FILES = {
    f"notes/{day(21)}.md": f"""# {day(21)}

Called my sister Lena. Her birthday is on 4 October, and she has wanted to try a pottery
class for ages, so a voucher for the ceramics studio on Linden Street is the plan.""",
    f"notes/{day(9)}.md": f"""# {day(9)}

Ran 8 km along the canal, the longest since my knee got better. The physio said to add at
most 1 km a week until the half marathon in spring.""",
    f"notes/{day(3)}.md": f"""# {day(3)}

Started reading "The Overstory". Book club meets on the last Thursday of the month; I host
next time, so I need to book the back room at Café Mira.""",
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
    print(f"[setup] Wrote {len(SAMPLE_FILES)} sample notes to sample/notes/.")
    print(
        "[setup] Now run: uv run python "
        "examples/cookbooks/self_hosted_companion/self_hosted_companion.py --sample"
    )
