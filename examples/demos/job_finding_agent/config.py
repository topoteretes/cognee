"""Configuration for the job-finding demo."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / "data" / "mock_jobs.json"
SKILL_FILE = ROOT_DIR / "skills" / "skill.md"
CV_FILE = ROOT_DIR / "data" / "cv.md"

APPLICANT_DATASET_NAME = "applicant_data"
DATASET_NAME = "job_finding_agent_demo"
SESSION_ID = "job_finding_session"
MAX_ITERATIONS = 10
