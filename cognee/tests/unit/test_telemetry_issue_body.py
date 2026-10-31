"""Exercise the telemetry Action's issue-body guard; no Cognee runtime is needed."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / ".github/scripts/telemetry_issue_body.py"

TITLE = "# Telemetry insights: 2026-09-27 — 1.6.0 error spike\n\n"
SIMPLE_HEADER = "## Simply put\n\n| # | Problem | Fix |\n|---|---|---|\n"
DETAIL_HEADER = "## Details\n\n| # | Observation | Analysis | Suggested fix |\n|---|---|---|---|\n"

PROBLEM = (
    "One in five of the newest installs failed at building memory yesterday, up from one in thirty."
)
PLAIN_FIX = "Record the setup on failed runs too, then check whether the new group of installs is the one failing."
ANALYSIS = "Errors carry no provider: provider_stack_daily counts Completed only (run_tasks_with_telemetry.py:48)."
FIX = "Count started/errored runs in provider_stack_daily; confirm one stack carries most 1.6.0 errors."


def simple_row(number, problem=PROBLEM):
    return f"| {number} | {problem} | {PLAIN_FIX} |\n"


def detail_row(number, observation="1.6.0 error rate 22.6% vs 3.5% (fleet 11.7%)"):
    return f"| {number} | {observation} | {ANALYSIS} | {FIX} |\n"


def issue(*numbers, **overrides):
    numbers = numbers or (1,)
    simple = "".join(simple_row(n, overrides.get(f"problem{n}", PROBLEM)) for n in numbers)
    detail = "".join(
        detail_row(n, overrides.get(f"obs{n}", "1.6.0 error rate 22.6% vs 3.5% (fleet 11.7%)"))
        for n in numbers
    )
    return TITLE + SIMPLE_HEADER + simple + "\n" + DETAIL_HEADER + detail


class TelemetryIssueBodyTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("telemetry_issue_body", SCRIPT)
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)

    def test_title_tables_and_run_link(self):
        title, body, trimmed = self.guard.build(issue(1), "https://example.test/run/1")
        self.assertEqual(title, "Telemetry insights: 2026-09-27 — 1.6.0 error spike")
        self.assertTrue(body.startswith("## Simply put\n\n| # | Problem | Fix |"))
        self.assertIn("\n\n## Details\n\n| # | Observation | Analysis | Suggested fix |", body)
        self.assertIn(f"| 1 | {PROBLEM} | {PLAIN_FIX} |", body)
        self.assertIn("| 1 | 1.6.0 error rate 22.6%", body)
        self.assertIn("[workflow run](https://example.test/run/1)", body)
        self.assertFalse(trimmed)

    def test_multiple_findings_are_kept_in_order_in_both_tables(self):
        simple, detail = self.guard.parse_tables(
            self.guard.split_title(issue(1, 2, 3, problem2="second", obs3="third"))[1]
        )
        self.assertEqual([r[0] for r in simple], ["1", "2", "3"])
        self.assertEqual(simple[1][1], "second")
        self.assertEqual(detail[2][1], "third")

    def test_wrong_header_or_heading_fails(self):
        with self.assertRaisesRegex(self.guard.IssueFormatError, "Details header"):
            self.guard.build(issue(1).replace("Suggested fix", "Fix"), None)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "expected heading"):
            self.guard.build(issue(1).replace("## Simply put", "## Summary"), None)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "expected heading"):
            self.guard.build(TITLE + DETAIL_HEADER + detail_row(1), None)

    def test_empty_cell_and_text_outside_tables_fail(self):
        empty = issue(1).replace(f"| {PROBLEM} |", "|  |")
        with self.assertRaisesRegex(self.guard.IssueFormatError, "empty cell"):
            self.guard.build(empty, None)
        prose = issue(1) + "\n## Watching\nsomething near threshold\n"
        with self.assertRaisesRegex(self.guard.IssueFormatError, "outside the tables"):
            self.guard.build(prose, None)

    def test_tables_with_different_findings_fail(self):
        text = issue(1) + detail_row(2)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "different findings"):
            self.guard.build(text, None)

    def test_missing_or_foreign_title_fails(self):
        with self.assertRaisesRegex(self.guard.IssueFormatError, "H1 title"):
            self.guard.build(issue(1)[len(TITLE) :], None)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "must start with"):
            self.guard.build("# Weekly digest\n\n" + issue(1)[len(TITLE) :], None)

    def test_findings_beyond_max_are_dropped_from_both_tables(self):
        text = issue(1, 2, 3, 4, 5, problem4="plain four", obs4="detail four")
        _, body, trimmed = self.guard.build(text, None)
        self.assertTrue(trimmed)
        self.assertIn("| 3 |", body)
        self.assertNotIn("plain four", body)
        self.assertNotIn("detail four", body)
        self.assertTrue(body.endswith(self.guard.TRIM_NOTE))

    def test_long_body_drops_whole_findings(self):
        one = self.guard.render(self.guard.parse_tables(issue(1)[len(TITLE) :]))
        limit = len(one) + len(self.guard.TRIM_NOTE) + 10
        _, body, trimmed = self.guard.build(issue(1, 2, 3), None, limit=limit)
        self.assertTrue(trimmed)
        self.assertEqual(body.count("| 1 |"), 2)
        self.assertNotIn("| 2 |", body)
        self.assertLessEqual(len(body), limit)

    def test_long_cell_is_cut_at_a_word_boundary_without_row_note(self):
        words = " ".join(f"word{i}" for i in range(60))
        _, body, trimmed = self.guard.build(issue(1, problem1=words), None)
        self.assertTrue(trimmed)
        self.assertNotIn(self.guard.TRIM_NOTE, body)  # nothing dropped, only clipped
        cell = self.guard.parse_tables(body)[0][0][1]
        self.assertLessEqual(len(cell), self.guard.MAX_CELL_CHARS)
        self.assertTrue(cell.endswith("…"))
        self.assertRegex(cell[:-1], r"word\d+$")  # no half word before the ellipsis

    def test_br_is_flattened_to_one_line(self):
        _, body, trimmed = self.guard.build(
            issue(1, problem1="first point<br>second point<br/>third"), None
        )
        self.assertTrue(trimmed)
        self.assertNotIn("<br", body)
        self.assertIn("| 1 | first point second point third |", body)

    def test_short_tables_are_not_trimmed(self):
        tables = self.guard.parse_tables(issue(1)[len(TITLE) :])
        body, trimmed = self.guard.trim(tables, limit=10_000)
        self.assertEqual(body, self.guard.render(tables))
        self.assertFalse(trimmed)


if __name__ == "__main__":
    unittest.main()
