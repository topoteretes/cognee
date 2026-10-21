"""Exercise the weekly-ticket Action's issue-body guard; no Cognee runtime is needed."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/weekly_tickets/issue_body.py"

SIMPLE_HEADER = "## Simply put\n\n| # | Problem | Fix |\n|---|---|---|\n"
DETAIL_HEADER = "## Details\n\n| # | Evidence | Root cause | Fix shape |\n|---|---|---|---|\n"

PROBLEM = "About one in six questions came from people whose local model setup fell back to OpenAI."
PLAIN_FIX = (
    "Refuse to start with a clear message when only one provider is set; watch the count drop."
)
EVIDENCE = "41 of 260 conversations in 'llm / model config'; 9 error reports mention an OpenAI 401."
ROOT_CAUSE = (
    "Embedding provider defaults to OpenAI when only LLM_PROVIDER is set (embeddings/config.py:88)."
)
FIX_SHAPE = "Raise a configuration error naming both providers when exactly one is set."


def simple_row(number, problem=PROBLEM):
    return f"| {number} | {problem} | {PLAIN_FIX} |\n"


def detail_row(number, evidence=EVIDENCE):
    return f"| {number} | {evidence} | {ROOT_CAUSE} | {FIX_SHAPE} |\n"


def issue(*numbers, **overrides):
    numbers = numbers or (1,)
    simple = "".join(simple_row(n, overrides.get(f"problem{n}", PROBLEM)) for n in numbers)
    detail = "".join(detail_row(n, overrides.get(f"evidence{n}", EVIDENCE)) for n in numbers)
    return SIMPLE_HEADER + simple + "\n" + DETAIL_HEADER + detail


class WeeklyTicketsIssueBodyTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("issue_body", SCRIPT)
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)

    def test_tables_and_run_link(self):
        body, trimmed = self.guard.build(issue(1), "https://example.test/run/1")
        self.assertTrue(body.startswith("## Simply put\n\n| # | Problem | Fix |"))
        self.assertIn("\n\n## Details\n\n| # | Evidence | Root cause | Fix shape |", body)
        self.assertIn(f"| 1 | {PROBLEM} | {PLAIN_FIX} |", body)
        self.assertIn(f"| 1 | {EVIDENCE} | {ROOT_CAUSE} | {FIX_SHAPE} |", body)
        self.assertIn("[workflow run](https://example.test/run/1)", body)
        self.assertFalse(trimmed)

    def test_multiple_proposals_are_kept_in_order_in_both_tables(self):
        simple, detail = self.guard.parse_tables(
            issue(1, 2, 3, problem2="second", evidence3="third")
        )
        self.assertEqual([r[0] for r in simple], ["1", "2", "3"])
        self.assertEqual(simple[1][1], "second")
        self.assertEqual(detail[2][1], "third")

    def test_wrong_header_or_heading_fails(self):
        with self.assertRaisesRegex(self.guard.IssueFormatError, "Details header"):
            self.guard.build(issue(1).replace("Fix shape", "Fix"), None)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "expected heading"):
            self.guard.build(issue(1).replace("## Simply put", "## Summary"), None)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "expected heading"):
            self.guard.build(DETAIL_HEADER + detail_row(1), None)

    def test_old_report_format_fails(self):
        report = "## Proposed tickets\n\n### 1. Fail fast on half-configured providers\n"
        with self.assertRaisesRegex(self.guard.IssueFormatError, "expected heading"):
            self.guard.build(report, None)

    def test_empty_cell_and_text_outside_tables_fail(self):
        empty = issue(1).replace(f"| {PROBLEM} |", "|  |")
        with self.assertRaisesRegex(self.guard.IssueFormatError, "empty cell"):
            self.guard.build(empty, None)
        prose = issue(1) + "\n## Not filed\n- something that was user error\n"
        with self.assertRaisesRegex(self.guard.IssueFormatError, "outside the tables"):
            self.guard.build(prose, None)

    def test_tables_with_different_proposals_fail(self):
        text = issue(1) + detail_row(2)
        with self.assertRaisesRegex(self.guard.IssueFormatError, "different proposals"):
            self.guard.build(text, None)

    def test_proposals_beyond_max_are_dropped_from_both_tables(self):
        text = issue(1, 2, 3, 4, 5, problem4="plain four", evidence4="detail four")
        body, trimmed = self.guard.build(text, None)
        self.assertTrue(trimmed)
        self.assertIn("| 3 |", body)
        self.assertNotIn("plain four", body)
        self.assertNotIn("detail four", body)
        self.assertTrue(body.endswith(self.guard.TRIM_NOTE))

    def test_long_body_drops_whole_proposals(self):
        one = self.guard.render(self.guard.parse_tables(issue(1)))
        limit = len(one) + len(self.guard.TRIM_NOTE) + 10
        body, trimmed = self.guard.build(issue(1, 2, 3), None, limit=limit)
        self.assertTrue(trimmed)
        self.assertEqual(body.count("| 1 |"), 2)
        self.assertNotIn("| 2 |", body)
        self.assertLessEqual(len(body), limit)

    def test_long_cell_is_cut_at_a_word_boundary_without_row_note(self):
        words = " ".join(f"word{i}" for i in range(60))
        body, trimmed = self.guard.build(issue(1, problem1=words), None)
        self.assertTrue(trimmed)
        self.assertNotIn(self.guard.TRIM_NOTE, body)  # nothing dropped, only clipped
        cell = self.guard.parse_tables(body)[0][0][1]
        self.assertLessEqual(len(cell), self.guard.MAX_CELL_CHARS)
        self.assertTrue(cell.endswith("…"))
        self.assertRegex(cell[:-1], r"word\d+$")  # no half word before the ellipsis

    def test_br_is_flattened_to_one_line(self):
        body, trimmed = self.guard.build(
            issue(1, problem1="first point<br>second point<br/>third"), None
        )
        self.assertTrue(trimmed)
        self.assertNotIn("<br", body)
        self.assertIn("| 1 | first point second point third |", body)

    def test_short_tables_are_not_trimmed(self):
        tables = self.guard.parse_tables(issue(1))
        body, trimmed = self.guard.trim(tables, limit=10_000)
        self.assertEqual(body, self.guard.render(tables))
        self.assertFalse(trimmed)


if __name__ == "__main__":
    unittest.main()
