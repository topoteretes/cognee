"""Unit tests for Markdown task-list data-source connector."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cognee.tasks.ingestion.connectors.markdown_task_list import (
    MarkdownTaskPathNotFoundError,
    markdown_task_list_source,
    parse_markdown_tasks,
)


class TestMarkdownTaskListConnector(unittest.TestCase):
    def test_parse_markdown_tasks(self):
        content = (
            "# Sprint 1\n\n"
            "- [ ] Task Alpha\n"
            "- [x] Task Beta\n"
            "Just some notes\n\n"
            "## Sub-section\n"
            "* [ ] Task Gamma\n"
            "+ [X] Task Delta\n"
        )
        tasks = parse_markdown_tasks(content, rel_path="docs/todo.md")
        self.assertEqual(len(tasks), 4)

        # Task Alpha
        self.assertEqual(tasks[0]["title"], "[ ] Task Alpha (todo.md)")
        self.assertFalse(tasks[0]["completed"])
        self.assertEqual(tasks[0]["section"], "Sprint 1")
        self.assertEqual(tasks[0]["line"], 3)
        self.assertIn("Status: Pending", tasks[0]["content"])

        # Task Beta
        self.assertEqual(tasks[1]["title"], "[x] Task Beta (todo.md)")
        self.assertTrue(tasks[1]["completed"])
        self.assertEqual(tasks[1]["section"], "Sprint 1")
        self.assertEqual(tasks[1]["line"], 4)
        self.assertIn("Status: Completed", tasks[1]["content"])

        # Task Gamma
        self.assertEqual(tasks[2]["section"], "Sub-section")
        self.assertFalse(tasks[2]["completed"])

        # Task Delta
        self.assertEqual(tasks[3]["section"], "Sub-section")
        self.assertTrue(tasks[3]["completed"])

    def test_parse_bom_and_crlf(self):
        content = "\ufeff# BOM Header\r\n- [ ] Clean task\r\n- [x] Done task\r\n"
        tasks = parse_markdown_tasks(content, rel_path="notes.md")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["section"], "BOM Header")
        self.assertEqual(tasks[0]["title"], "[ ] Clean task (notes.md)")

    def test_missing_path_raises_error(self):
        with self.assertRaises(MarkdownTaskPathNotFoundError):
            markdown_task_list_source("/non/existent/path/here")

    def test_source_yields_rows_and_tombstones(self):
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            md_file = tmppath / "tasks.md"
            md_file.write_text(
                "# Project\n- [ ] Setup CI\n- [x] Initial commit\n", encoding="utf-8"
            )

            # First sync run
            state = {}
            stats = {"synced": 0, "deleted": 0}
            from cognee.tasks.ingestion.connectors.markdown_task_list import _iter_task_rows

            rows = list(
                _iter_task_rows(tmppath, state=state, stats=stats, node_set="markdown_tasks")
            )
            self.assertEqual(len(rows), 2)
            self.assertFalse(rows[0]["_deleted"])
            self.assertFalse(rows[1]["_deleted"])
            self.assertEqual(rows[0]["cognee_node_set"], "markdown_tasks")

            # Update file by removing one task
            md_file.write_text("# Project\n- [ ] Setup CI\n", encoding="utf-8")
            rows2 = list(
                _iter_task_rows(tmppath, state=state, stats=stats, node_set="markdown_tasks")
            )
            # Second run yields 1 tombstone for the deleted item
            deleted_rows = [r for r in rows2 if r["_deleted"]]
            self.assertEqual(len(deleted_rows), 1)
            self.assertIn("tasks.md:3", deleted_rows[0]["id"])
            self.assertEqual(stats["deleted"], 1)

            # Test resource wrapper creation
            source = markdown_task_list_source(tmppath, resource_name="test_res")
            res_rows = list(source)
            self.assertEqual(len(res_rows), 1)
            self.assertFalse(res_rows[0]["_deleted"])


if __name__ == "__main__":
    unittest.main()
