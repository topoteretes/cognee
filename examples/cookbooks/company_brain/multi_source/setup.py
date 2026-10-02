"""Set up the sample company, so the cookbook runs before you point it at your own data.

The sample is Acorn Analytics, a fictional company, in sample/: an HR and project database
(built here from schema.sql into company.db), a support ticket export and a folder of
meeting notes. Some people, projects and customers appear in all three, so the run shows
them merge into one node each. Only LLM_API_KEY is needed; no accounts or data of yours.

Run: uv run python examples/cookbooks/company_brain/multi_source/setup.py
"""

import sqlite3
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"
DATABASE = SAMPLE / "company.db"


def build_database() -> None:
    """Build company.db from schema.sql, replacing any earlier copy."""
    DATABASE.unlink(missing_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.executescript((SAMPLE / "schema.sql").read_text())
    connection.close()


if __name__ == "__main__":
    build_database()
    print("[setup] Built sample/company.db from sample/schema.sql.")
    print(
        "[setup] Now run: uv run python "
        "examples/cookbooks/company_brain/multi_source/company_brain.py --sample"
    )
