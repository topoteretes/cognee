"""CLI plumbing shared by ``search`` and ``recall`` for ``--query-type CODE``.

Three options ride on top of the ordinary search flags:

- ``--code-query JSON``: the structured ``code_query`` dict (operation and
  arguments) the SDK/API take, as a JSON string.
- ``--diagram {mermaid,dot}``: ask for the result rendered as a diagram too.
- ``--diagram-out PATH``: write the diagram somewhere useful — a ``.html``
  page that renders Mermaid in any browser, an ``.svg``/``.png``/``.pdf``
  produced by Graphviz for DOT, or the raw source for any other extension.

The pretty printer shows the structured result and the diagram source in a
```` ```mermaid ```` fence, which GitHub, most docs sites and chat clients
render as a picture.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
from typing import Any, Iterator, Optional

import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandInnerException

DIAGRAM_FORMAT_CHOICES = ["mermaid", "dot"]
CODE_OPERATIONS = (
    "query_facts",
    "explore",
    "traverse",
    "find_path",
    "impact_analysis",
    "insights",
    "architecture",
    "delta",
)
_DOT_EXTENSIONS = {".dot", ".gv"}
_GRAPHVIZ_RENDER_EXTENSIONS = {".svg", ".png", ".pdf"}
_MERMAID_CDN = "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.15.0/mermaid.min.js"

_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 1.5rem; font-family: system-ui, sans-serif; background: #fff; color: #222; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem; }}
  .mermaid {{ overflow: auto; }}
  details {{ margin-top: 1.5rem; }}
  pre {{ background: #f6f8fa; padding: 1rem; overflow: auto; font-size: 0.85rem; }}
</style>
<script src="{cdn}"></script>
</head>
<body>
<h1>{title}</h1>
<pre class="mermaid">
{source}
</pre>
<details><summary>Mermaid source</summary><pre>{source}</pre></details>
<script>mermaid.initialize({{ startOnLoad: true, securityLevel: "strict" }});</script>
</body>
</html>
"""


def add_code_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the CODE-only options on a search/recall parser."""
    parser.add_argument(
        "--code-query",
        metavar="JSON",
        default=None,
        help=(
            "Structured code-graph query for --query-type CODE, as JSON: "
            '\'{"operation": "architecture"}\', \'{"operation": "impact_analysis", '
            '"name": "pkg.Symbol", "max_depth": 3}\'. Operations: ' + ", ".join(CODE_OPERATIONS)
        ),
    )
    parser.add_argument(
        "--diagram",
        choices=DIAGRAM_FORMAT_CHOICES,
        default=None,
        help=(
            "Also render the CODE result as a diagram (Mermaid or Graphviz DOT source). "
            "The architecture operation draws Mermaid by default."
        ),
    )
    parser.add_argument(
        "--diagram-out",
        metavar="PATH",
        default=None,
        help=(
            "Write the diagram to PATH: .html renders Mermaid in a browser, .svg/.png/.pdf "
            "run Graphviz `dot` on a DOT diagram, anything else gets the raw source. "
            "Implies --diagram (format inferred from the extension)."
        ),
    )


def build_code_query(args: argparse.Namespace, query_type: Optional[str]) -> Optional[dict]:
    """Assemble ``code_query`` from --code-query/--diagram/--diagram-out, or None."""
    raw = getattr(args, "code_query", None)
    diagram = getattr(args, "diagram", None)
    diagram_out = getattr(args, "diagram_out", None)
    if raw is None and diagram is None and diagram_out is None:
        return None
    if query_type != "CODE":
        raise CliCommandInnerException(
            "--code-query, --diagram and --diagram-out require --query-type CODE."
        )

    code_query: dict[str, Any] = {}
    if raw is not None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CliCommandInnerException(f"--code-query is not valid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise CliCommandInnerException("--code-query must be a JSON object.")
        code_query.update(parsed)

    if diagram is not None:
        code_query["diagram"] = diagram
    elif diagram_out is not None and "diagram" not in code_query:
        code_query["diagram"] = infer_diagram_format(diagram_out)
    return code_query


def infer_diagram_format(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    if extension in _DOT_EXTENSIONS or extension in _GRAPHVIZ_RENDER_EXTENSIONS:
        return "dot"
    return "mermaid"


def iter_code_results(results: Any) -> Iterator[dict]:
    """Yield every CODE operation result inside a search()/recall() response.

    search() wraps results per dataset ({dataset_id, dataset_name,
    search_result: [...]}) and recall() returns normalized items whose ``raw``
    field keeps the payload; both are walked recursively (pydantic models via
    model_dump) for dicts carrying an ``operation`` key.
    """
    if hasattr(results, "model_dump") and callable(results.model_dump):
        results = results.model_dump(mode="python")
    if isinstance(results, dict):
        if "operation" in results and isinstance(results.get("operation"), str):
            yield results
            return
        for value in results.values():
            yield from iter_code_results(value)
    elif isinstance(results, (list, tuple)):
        for item in results:
            yield from iter_code_results(item)


def first_diagram(results: Any) -> Optional[dict]:
    for result in iter_code_results(results):
        diagram = result.get("diagram")
        if isinstance(diagram, dict) and diagram.get("source"):
            return diagram
    return None


def _summary_line(result: dict) -> str:
    operation = result.get("operation")
    parts = []
    if "total" in result:
        parts.append(f"{result['total']} total")
    stats = result.get("stats")
    if isinstance(stats, dict):
        for key in ("nodes_shown", "nodes_visited", "edges_shown", "edges_traversed", "truncated"):
            if key in stats:
                parts.append(f"{key}={stats[key]}")
    if "found" in result:
        parts.append("path found" if result["found"] else "no path")
    if "summary" in result:
        parts.append(str(result["summary"]))
    return f"{operation}" + (f" ({', '.join(parts)})" if parts else "")


def print_code_results(results: Any) -> bool:
    """Pretty-print CODE results: structured payload, then the diagram fenced.

    Returns False when no CODE result was found so the caller can fall back
    to its generic formatter.
    """
    code_results = list(iter_code_results(results))
    if not code_results:
        return False
    for index, result in enumerate(code_results, 1):
        fmt.echo(f"{fmt.bold(f'Result {index}:')} {_summary_line(result)}")
        payload = {key: value for key, value in result.items() if key != "diagram"}
        fmt.echo(json.dumps(payload, indent=2, default=str))
        diagram = result.get("diagram")
        if isinstance(diagram, dict):
            if diagram.get("source"):
                diagram_format = diagram.get("format")
                heading = fmt.bold(f"Diagram ({diagram_format}):")
                fmt.echo()
                fmt.echo(
                    f"{heading} {diagram.get('nodes')} node(s), {diagram.get('edges')} edge(s)"
                )
                fmt.echo(f"```{diagram_format}")
                fmt.echo(str(diagram["source"]).rstrip("\n"))
                fmt.echo("```")
            elif diagram.get("note"):
                fmt.echo(f"Diagram: {diagram['note']}")
        fmt.echo()
    return True


def write_diagram(results: Any, path: str) -> str:
    """Write the first diagram in results to path; returns the written path.

    ``.html`` wraps Mermaid source in a page that renders it (Mermaid loaded
    from a CDN when opened). ``.svg``/``.png``/``.pdf`` render DOT through the
    Graphviz ``dot`` binary. Everything else receives the raw source.
    """
    diagram = first_diagram(results)
    if diagram is None:
        raise CliCommandInnerException(
            "No diagram to write: the result carries none. Add --diagram or use an operation "
            "that returns a graph (delta does not)."
        )
    diagram_format = diagram.get("format")
    source = str(diagram["source"])
    extension = os.path.splitext(path)[1].lower()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if extension in {".html", ".htm"}:
        if diagram_format != "mermaid":
            raise CliCommandInnerException(
                "An .html diagram page needs the Mermaid format; use --diagram mermaid, or "
                "write DOT to .svg/.png/.pdf (Graphviz) or .dot."
            )
        title = _mermaid_title(source) or "cognee code graph"
        with open(path, "w", encoding="utf-8") as page:
            page.write(
                _HTML_PAGE.format(
                    title=html.escape(title), cdn=_MERMAID_CDN, source=html.escape(source)
                )
            )
        return path

    if extension in _GRAPHVIZ_RENDER_EXTENSIONS:
        if diagram_format != "dot":
            raise CliCommandInnerException(
                f"Rendering to {extension} needs the DOT format (Graphviz); use --diagram dot, "
                "or write Mermaid to .html."
            )
        dot_binary = shutil.which("dot")
        if dot_binary is None:
            raise CliCommandInnerException(
                f"Rendering to {extension} needs Graphviz (`dot`) on PATH; install it or write "
                "the raw source to a .dot file."
            )
        completed = subprocess.run(
            [dot_binary, f"-T{extension[1:]}", "-o", path],
            input=source.encode("utf-8"),
            capture_output=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise CliCommandInnerException(
                f"Graphviz failed ({completed.returncode}): "
                f"{completed.stderr.decode(errors='replace')[-500:]}"
            )
        return path

    with open(path, "w", encoding="utf-8") as raw_file:
        raw_file.write(source)
    return path


def _mermaid_title(source: str) -> Optional[str]:
    lines = source.splitlines()
    if len(lines) >= 3 and lines[0] == "---":
        for line in lines[1:]:
            if line == "---":
                break
            if line.startswith("title:"):
                title = line[len("title:") :].strip()
                if len(title) >= 2 and title[0] == title[-1] == '"':
                    title = title[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                return title or None
    return None


def handle_diagram_out(results: Any, args: argparse.Namespace) -> None:
    """Honor --diagram-out after the results have been printed."""
    path = getattr(args, "diagram_out", None)
    if not path:
        return
    written = write_diagram(results, path)
    fmt.success(f"Diagram written to {written}")
    if os.path.splitext(written)[1].lower() in {".html", ".htm"}:
        fmt.echo("Open it in a browser to see the rendered architecture.")
