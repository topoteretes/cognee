"""Exercise the workflow stamp commands and warehouse views without cloud credentials."""

import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = ROOT / ".github/workflows"
CASES = []
for filename in (
    "performance_report.yml",
    "performance_report_cloud.yml",
    "performance_report_rust.yml",
):
    workflow = yaml.safe_load((WORKFLOWS / filename).read_text())
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if step.get("name") == "Stamp run provenance into the report":
                CASES.append(pytest.param(filename, step, id=f"{filename}-{job_name}"))


def git(repo, *args, env=None):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, env=env, stderr=subprocess.PIPE
    ).strip()


def checkout(repo, date):
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture", env=env)
    git(repo, "checkout", "--detach", "HEAD")
    return git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("filename,step", CASES)
def test_stamp_uses_checked_out_commit_and_preserves_results(tmp_path, filename, step):
    workflow_date = "2026-08-01T12:00:00+02:00"
    rust_date = "2026-08-02T10:00:00+00:00"
    workflow_sha = checkout(tmp_path, workflow_date)
    rust_sha = checkout(tmp_path / "cognee-rs", rust_date)
    report = tmp_path / "report.json"
    original = {"num_runs": 1, "stats": {"latency": {"p50": 1.25}}}
    report.write_text(json.dumps(original))
    env = {
        **os.environ,
        "JSON_PATH": report.as_posix(),
        "BRANCH": "dev",
        "REPOSITORY": "topoteretes/cognee",
        "RUN_ID": "123",
        "RUN_ATTEMPT": "2",
        "EVENT": "workflow_dispatch",
        # Deliberately different: event context must not override the checkout.
        "SHA": "0" * 40,
        "GITHUB_SHA": "0" * 40,
    }
    bash = shutil.which("bash")
    if os.name == "nt":
        # PATH may resolve bash to the WSL launcher, which cannot execute
        # these GitHub-hosted Windows tests without an installed distro.
        git_executable = Path(shutil.which("git"))
        bash = next(
            (
                parent / "bin" / "bash.exe"
                for parent in git_executable.parents
                if (parent / "bin" / "bash.exe").is_file()
            ),
            None,
        )
        assert bash is not None, "Git for Windows Bash is required for workflow tests"
        # The unit runner intentionally removes coreutils from PATH. Restore
        # them only for this child process, which executes a Bash workflow.
        env["PATH"] = os.pathsep.join(
            [str(bash.parent), str(bash.parent.parent / "usr" / "bin"), env["PATH"]]
        )
    assert bash is not None, "Bash is required for workflow tests"
    subprocess.run([str(bash), "-c", step["run"]], cwd=tmp_path, env=env, check=True)
    result = json.loads(report.read_text())
    rust = filename == "performance_report_rust.yml"
    assert result["git_sha"] == (rust_sha if rust else workflow_sha)
    # Git versions spell UTC as either Z or +00:00. Both represent the
    # same commit time; compare instants rather than serialization choices.
    actual_time = datetime.fromisoformat(result["commit_timestamp"].replace("Z", "+00:00"))
    assert actual_time == datetime.fromisoformat(rust_date if rust else workflow_date)
    assert result["git_repository"] == ("topoteretes/cognee-rs" if rust else "topoteretes/cognee")
    if rust:
        assert result["workflow_git_sha"] == workflow_sha
    assert result["branch"] == "dev"
    assert result["run_id"] == "123"
    assert result["run_attempt"] == "2"
    assert result["event"] == "workflow_dispatch"
    assert all(result[key] == value for key, value in original.items())


def test_warehouse_views_expose_provenance_and_preserve_historical_rows():
    duckdb = pytest.importorskip("duckdb")
    spec = importlib.util.spec_from_file_location(
        "perf_etl", ROOT / ".github/scripts/motherduck_nightly_etl.py"
    )
    etl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(etl)
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE SCHEMA nightly")
        con.execute(
            "CREATE TABLE nightly.raw_perf_reports "
            "(s3_key VARCHAR, uploaded_at TIMESTAMP, size_bytes BIGINT, report JSON)"
        )
        base = {"num_runs": 1, "succeeded": 1, "failed": 0, "stats": {"latency": {"p50": 1.25}}}
        provenance = {
            "git_sha": "a" * 40,
            "commit_timestamp": "2026-08-01T12:00:00+02:00",
            "git_repository": "topoteretes/cognee-rs",
            "workflow_git_sha": "b" * 40,
        }
        for day, extra in enumerate(
            [{}, {"git_sha": "c" * 40}, provenance, {"commit_timestamp": "invalid"}], start=1
        ):
            key = f"s3://bucket/performance_results/rust_file_based/small/mock_llm_2026-08-0{day}_12-00-00Z.json"
            con.execute(
                "INSERT INTO nightly.raw_perf_reports VALUES (?, current_timestamp, 1, ?)",
                [key, json.dumps({**base, **extra})],
            )
        # The same CREATE OR REPLACE pass used by the ETL upgrades existing views.
        for _ in range(2):
            for name, sql in etl.VIEWS.items():
                con.execute(f"CREATE OR REPLACE VIEW nightly.{name} AS {sql.format(t='nightly')}")
        for view in ("v_perf_runs", "v_perf_metrics"):
            rows = con.execute(
                f"SELECT git_sha, commit_timestamp, git_repository, workflow_git_sha "
                f"FROM nightly.{view} ORDER BY run_ts"
            ).fetchall()
            assert len(rows) == 4
            assert rows[0] == (None, None, None, None)
            assert rows[1] == ("c" * 40, None, None, None)
            assert rows[2] == (
                "a" * 40,
                datetime.fromisoformat(provenance["commit_timestamp"]),
                "topoteretes/cognee-rs",
                "b" * 40,
            )
            assert rows[3] == (None, None, None, None)
        assert con.execute("SELECT count(*) FROM nightly.v_perf_regression").fetchone()[0] == 1
    finally:
        con.close()
