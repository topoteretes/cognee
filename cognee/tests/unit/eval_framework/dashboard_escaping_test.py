"""Regression tests: eval dashboards must HTML-escape benchmark text.

``generate_details_html`` interpolates per-item benchmark fields (question,
answer, golden_answer, reason) straight into an HTML table. Without escaping,
any ``<`` / ``&`` / tag-like content in the model or golden answer corrupts the
table layout (and is a stored-HTML-injection vector when the dashboard is
opened). Both the active dashboard (``metrics_dashboard``) and its analysis
twin (``analysis.dashboard_generator``) must escape these values.
"""

import unittest

from cognee.eval_framework import metrics_dashboard
from cognee.eval_framework.analysis import dashboard_generator

MALICIOUS = 'a < b </td><script>alert("x")</script> & "q"'


def _detail_data():
    return [
        {
            "question": MALICIOUS,
            "answer": "answer",
            "golden_answer": "golden",
            "metrics": {"accuracy": {"score": 1.0, "reason": "ok"}},
        }
    ]


class TestDashboardEscaping(unittest.TestCase):
    def _assert_escaped(self, rendered: str):
        # The raw payload must not survive; its escaped form must be present.
        self.assertNotIn("<script>alert", rendered)
        self.assertNotIn("a < b </td>", rendered)
        self.assertIn("&lt;", rendered)
        self.assertIn("&amp;", rendered)

    def test_metrics_dashboard_escapes_details(self):
        rendered = "".join(metrics_dashboard.generate_details_html(_detail_data()))
        self._assert_escaped(rendered)

    def test_dashboard_generator_escapes_details(self):
        rendered = "".join(dashboard_generator.generate_details_html(_detail_data()))
        self._assert_escaped(rendered)

    def test_benchmark_name_is_escaped_in_template(self):
        for module in (metrics_dashboard, dashboard_generator):
            # figures[-1] is indexed by the template, so pass a non-empty list.
            rendered = module.get_dashboard_html_template(
                ["<div>f1</div>", "<div>f2</div>"], [], "<b>bench</b>"
            )
            self.assertNotIn("<b>bench</b>", rendered)
            self.assertIn("&lt;b&gt;bench&lt;/b&gt;", rendered)


if __name__ == "__main__":
    unittest.main()
