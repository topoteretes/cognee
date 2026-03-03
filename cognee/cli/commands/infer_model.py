"""Interactive CLI tool that infers a graph model from a user's problem description and data.

Uses an LLM to propose entity types, fields, and relationships, then iterates
through clarifying questions and feedback until the user is satisfied.  Outputs
both a JSON schema (usable with ``graph_schema_to_graph_model``) and a
production-ready ``models.py`` file with DataPoint inheritance.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import textwrap
from typing import List, Optional

from pydantic import BaseModel, Field

from cognee.cli import DEFAULT_DOCS_URL, SupportsCliCommand
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
import cognee.cli.echo as fmt


# ---------------------------------------------------------------------------
# 1. Intermediate Pydantic models for structuring LLM proposals
# ---------------------------------------------------------------------------


class FieldProposal(BaseModel):
    """A single field on an entity."""

    name: str = Field(description="snake_case field name")
    type: str = Field(description="Python type: str, int, float, bool")
    default: str = Field(default="", description='Default value as string, e.g. "" or "0"')
    description: str = Field(default="", description="Short explanation of the field")


class RelationshipProposal(BaseModel):
    """A typed relationship from one entity to another."""

    field_name: str = Field(description="snake_case field name on the source entity")
    target_entity: str = Field(description="PascalCase name of the target entity")
    cardinality: str = Field(description='"one" or "many"')


class EntityProposal(BaseModel):
    """One entity (node type) in the graph model."""

    name: str = Field(description="PascalCase class name, e.g. Supplier")
    description: str = Field(description="One-sentence description of what this entity represents")
    fields: List[FieldProposal] = Field(default_factory=list)
    relationships: List[RelationshipProposal] = Field(default_factory=list)
    index_fields: List[str] = Field(
        default_factory=list,
        description="Field names whose values should be embedded for vector search",
    )


class GraphModelProposal(BaseModel):
    """Complete graph model proposal returned by the LLM."""

    summary: str = Field(description="Brief summary of the domain the model covers")
    entities: List[EntityProposal] = Field(default_factory=list)
    root_entity_name: str = Field(
        description="PascalCase name of the root 'Context' entity that groups all others"
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Questions the LLM wants to ask to improve the model",
    )


# ---------------------------------------------------------------------------
# 2. Data scanner
# ---------------------------------------------------------------------------

_MAX_TEXT_PREVIEW = 500
_MAX_CSV_ROWS = 3


def _scan_csv(path: str) -> str:
    """Return a summary of a CSV file: headers + first few rows."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            content = fh.read()
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            if i >= _MAX_CSV_ROWS:
                break
            rows.append(row)
        lines = [f"CSV: {os.path.basename(path)}"]
        lines.append(f"  Columns ({len(headers)}): {', '.join(headers)}")
        for idx, row in enumerate(rows, 1):
            pairs = [f"{k}={v}" for k, v in row.items()]
            lines.append(f"  Row {idx}: {', '.join(pairs)}")
        return "\n".join(lines)
    except Exception as exc:
        return f"CSV: {os.path.basename(path)} — could not read: {exc}"


def _scan_text(path: str) -> str:
    """Return a preview of a text file."""
    try:
        with open(path, encoding="utf-8") as fh:
            preview = fh.read(_MAX_TEXT_PREVIEW)
        truncated = "..." if len(preview) == _MAX_TEXT_PREVIEW else ""
        return f"Text: {os.path.basename(path)}\n  {preview}{truncated}"
    except Exception as exc:
        return f"Text: {os.path.basename(path)} — could not read: {exc}"


def scan_files(paths: List[str]) -> str:
    """Scan a list of file paths and return a combined context string."""
    parts: list[str] = []
    for raw_path in paths:
        p = raw_path.strip()
        if not p:
            continue
        path = os.path.expanduser(p)
        if not os.path.isfile(path):
            parts.append(f"[not found: {p}]")
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            parts.append(_scan_csv(path))
        else:
            parts.append(_scan_text(path))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 3. System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a graph-model architect for the Cognee knowledge-graph platform.

    Your job is to design a **graph model** — a set of entity types (node types)
    with typed fields and relationships — that captures the user's problem domain.

    Rules:
    1. Each entity becomes a node type in a knowledge graph.
    2. Fields are simple scalar types: str, int, float, bool.
    3. Relationships are typed edges to other entities (cardinality: "one" or "many").
    4. Choose ``index_fields`` — the 1-3 string fields per entity most useful for
       semantic / vector search.  Prefer descriptive text fields.
    5. Always create a root "Context" entity whose fields are ``List[Entity]`` for
       every entity type (it serves as the extraction container).  Set its
       ``root_entity_name`` to this entity's name.
    6. Entity names must be PascalCase; field names must be snake_case.
    7. If the data or problem is ambiguous, include ``clarifying_questions``.
    8. When refining, preserve entities the user already approved and only change
       what the user asked to change.

    Respond ONLY with a valid ``GraphModelProposal`` JSON object.
""")


# ---------------------------------------------------------------------------
# 4. LLM helpers
# ---------------------------------------------------------------------------


async def _propose(conversation: list[dict]) -> GraphModelProposal:
    """Call the LLM and get a structured GraphModelProposal."""
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    messages_text = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in conversation)
    return await LLMGateway.acreate_structured_output(
        text_input=messages_text,
        system_prompt=SYSTEM_PROMPT,
        response_model=GraphModelProposal,
    )


# ---------------------------------------------------------------------------
# 5. Output generators
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "bool": "bool",
    "boolean": "bool",
}

_DEFAULT_FOR_TYPE = {
    "str": '""',
    "int": "0",
    "float": "0.0",
    "bool": "False",
}


def _py_type(raw: str) -> str:
    return _TYPE_MAP.get(raw.lower().strip(), "str")


def _py_default(field: FieldProposal) -> str:
    """Return a Python default-value expression for a field."""
    py_t = _py_type(field.type)
    if field.default:
        raw = field.default.strip()
        if py_t == "str":
            return f'"{raw}"' if not (raw.startswith('"') or raw.startswith("'")) else raw
        return raw
    return _DEFAULT_FOR_TYPE.get(py_t, '""')


def generate_json_schema(proposal: GraphModelProposal, output_path: str) -> None:
    """Build a Pydantic BaseModel from the proposal, convert to JSON schema, write to file.

    Tries ``graph_model_to_graph_schema`` (from commit 674a879) first for
    fidelity; falls back to plain ``model_json_schema`` when
    ``datamodel_code_generator`` is not installed.
    """
    from pydantic import create_model as pydantic_create_model

    entity_models: dict[str, type] = {}
    entity_names = {e.name for e in proposal.entities}

    for entity in proposal.entities:
        field_defs: dict = {}
        for f in entity.fields:
            py_t = _py_type(f.type)
            real_type = {"str": str, "int": int, "float": float, "bool": bool}.get(py_t, str)
            default = _py_default(f)
            try:
                evaluated_default = eval(default)  # noqa: S307
            except Exception:
                evaluated_default = ""
            field_defs[f.name] = (real_type, evaluated_default)

        for rel in entity.relationships:
            if rel.target_entity not in entity_names:
                continue
            if rel.cardinality == "many":
                field_defs[rel.field_name] = (Optional[List[str]], None)
            else:
                field_defs[rel.field_name] = (Optional[str], None)

        field_defs["metadata"] = (dict, {"index_fields": entity.index_fields})
        model = pydantic_create_model(entity.name, **field_defs)
        entity_models[entity.name] = model

    root_fields: dict = {}
    root_fields["summary"] = (str, "")
    for name, model in entity_models.items():
        list_name = _to_snake(name) + "s"
        root_fields[list_name] = (List[model], [])
    root_fields["metadata"] = (dict, {"index_fields": ["summary"]})

    root_model = pydantic_create_model(proposal.root_entity_name, **root_fields)

    try:
        from cognee.shared.graph_model_utils import graph_model_to_graph_schema

        schema = graph_model_to_graph_schema(root_model)
    except ImportError:
        schema = root_model.model_json_schema()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)


def generate_python_models(proposal: GraphModelProposal, output_path: str) -> None:
    """Generate a production-ready models.py with DataPoint subclasses."""

    lines: list[str] = []

    lines.append('"""Auto-generated graph model — edit freely to refine.')
    lines.append("")
    lines.append(f"Domain: {proposal.summary}")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import uuid")
    lines.append("from typing import List, Optional")
    lines.append("")
    lines.append("from pydantic import model_validator")
    lines.append("")
    lines.append("from cognee.infrastructure.engine import DataPoint")
    lines.append("")
    lines.append("")
    lines.append('_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")')
    lines.append("")
    lines.append("")

    # Base class with LLM-output sanitiser
    lines.append("class _BaseDomainDataPoint(DataPoint):")
    lines.append('    """Base class with LLM-output tolerance.')
    lines.append("")
    lines.append("    Converts non-UUID ids to deterministic UUIDs and coerces")
    lines.append('    null string fields to empty strings so validation passes."""')
    lines.append("")
    lines.append('    @model_validator(mode="before")')
    lines.append("    @classmethod")
    lines.append("    def _sanitize_llm_output(cls, data):")
    lines.append("        if not isinstance(data, dict):")
    lines.append("            return data")
    lines.append('        raw_id = data.get("id")')
    lines.append("        if isinstance(raw_id, str):")
    lines.append("            try:")
    lines.append("                uuid.UUID(raw_id)")
    lines.append("            except ValueError:")
    lines.append('                data["id"] = uuid.uuid5(_NS, f"{cls.__name__}:{raw_id}")')
    lines.append("        for field_name, field_info in cls.model_fields.items():")
    lines.append('            if field_name == "id":')
    lines.append("                continue")
    lines.append("            if data.get(field_name) is None and field_info.annotation is str:")
    lines.append('                data[field_name] = ""')
    lines.append("        return data")
    lines.append("")
    lines.append('    @model_validator(mode="after")')
    lines.append("    def _ensure_index_fields_non_empty(self):")
    lines.append('        for field_name in self.metadata.get("index_fields", []):')
    lines.append("            val = getattr(self, field_name, None)")
    lines.append("            if isinstance(val, str) and not val.strip():")
    lines.append('                object.__setattr__(self, field_name, "n/a")')
    lines.append("        return self")
    lines.append("")
    lines.append("")

    entity_names = {e.name for e in proposal.entities}
    needs_rebuild: list[str] = []

    for entity in proposal.entities:
        lines.append(f"class {entity.name}(_BaseDomainDataPoint):")
        if entity.description:
            lines.append(f'    """{entity.description}"""')
            lines.append("")

        for f in entity.fields:
            py_t = _py_type(f.type)
            default = _py_default(f)
            lines.append(f"    {f.name}: {py_t} = {default}")

        has_self_ref = False
        for rel in entity.relationships:
            if rel.target_entity not in entity_names:
                continue
            if rel.target_entity == entity.name:
                has_self_ref = True
            if rel.cardinality == "many":
                lines.append(f"    {rel.field_name}: Optional[List[{rel.target_entity}]] = None")
            else:
                lines.append(f"    {rel.field_name}: Optional[{rel.target_entity}] = None")

        idx = entity.index_fields or []
        idx_repr = ", ".join(f'"{f}"' for f in idx)
        lines.append("")
        lines.append(f'    metadata: dict = {{"index_fields": [{idx_repr}]}}')
        lines.append("")
        lines.append("")

        if has_self_ref:
            needs_rebuild.append(entity.name)

    for name in needs_rebuild:
        lines.append(f"{name}.model_rebuild()")
        lines.append("")

    # Root context class
    root_name = proposal.root_entity_name
    lines.append("")
    lines.append(f"class {root_name}(_BaseDomainDataPoint):")
    lines.append('    """Root model passed to ``cognify(graph_model=...)``.')
    lines.append("")
    lines.append("    Each chunk may populate only a subset of the lists below.")
    lines.append('    """')
    lines.append("")
    lines.append('    summary: str = ""')
    lines.append("")
    for entity in proposal.entities:
        list_name = _to_snake(entity.name) + "s"
        lines.append(f"    {list_name}: List[{entity.name}] = []")
    lines.append("")
    lines.append('    metadata: dict = {"index_fields": ["summary"]}')
    lines.append("")
    lines.append("")
    lines.append(f"{root_name}.model_rebuild()")
    lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _to_snake(name: str) -> str:
    """PascalCase -> snake_case."""
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# 6. Display helpers
# ---------------------------------------------------------------------------


def _display_proposal(proposal: GraphModelProposal) -> None:
    fmt.echo("")
    fmt.echo(fmt.bold("=== Proposed Graph Model ==="))
    fmt.echo(f"Summary: {proposal.summary}")
    fmt.echo(f"Root entity: {proposal.root_entity_name}")
    fmt.echo("")

    for entity in proposal.entities:
        fmt.echo(fmt.bold(f"  Entity: {entity.name}"))
        if entity.description:
            fmt.echo(f"    {entity.description}")

        if entity.fields:
            field_strs = [f"{f.name} ({f.type})" for f in entity.fields]
            fmt.echo(f"    Fields: {', '.join(field_strs)}")

        if entity.relationships:
            rel_strs = [
                f"{r.field_name} -> {r.target_entity} ({r.cardinality})"
                for r in entity.relationships
            ]
            fmt.echo(f"    Relationships: {', '.join(rel_strs)}")

        if entity.index_fields:
            fmt.echo(f"    Index fields: {', '.join(entity.index_fields)}")
        fmt.echo("")

    if proposal.clarifying_questions:
        fmt.echo(fmt.bold("  I have a few questions:"))
        for i, q in enumerate(proposal.clarifying_questions, 1):
            fmt.echo(f"    {i}. {q}")
        fmt.echo("")


# ---------------------------------------------------------------------------
# 7. Interactive CLI loop
# ---------------------------------------------------------------------------


async def _run_interactive(output_dir: str) -> None:
    fmt.echo("")
    fmt.echo(fmt.bold("Welcome! I'll help you design a graph model for your domain."))
    fmt.echo("")

    description = fmt.prompt("Describe your problem domain")

    fmt.echo("")
    file_input = fmt.prompt(
        "Do you have data files (CSV or text) I can look at? (comma-separated paths, or 'no')",
        default="no",
    )

    data_context = ""
    if file_input.strip().lower() not in ("no", "n", ""):
        paths = [p.strip() for p in file_input.split(",")]
        fmt.echo(f"\nScanning {len(paths)} file(s)...")
        data_context = scan_files(paths)
        fmt.echo(data_context)

    conversation: list[dict] = []
    user_msg = f"Problem domain:\n{description}"
    if data_context:
        user_msg += f"\n\nData files:\n{data_context}"
    conversation.append({"role": "user", "content": user_msg})

    while True:
        fmt.echo("\nGenerating model proposal...")
        try:
            proposal = await _propose(conversation)
        except Exception as exc:
            fmt.error(f"LLM call failed: {exc}")
            raise

        _display_proposal(proposal)

        if proposal.clarifying_questions:
            answers = fmt.prompt("Your answers (or press Enter to skip)")
            if answers.strip():
                conversation.append({"role": "assistant", "content": proposal.model_dump_json()})
                conversation.append(
                    {"role": "user", "content": f"Answers to your questions:\n{answers}"}
                )
                continue

        feedback = fmt.prompt(
            "Are you happy with this model? ('yes' to generate, or give feedback)"
        )
        if feedback.strip().lower() in ("yes", "y"):
            break

        conversation.append({"role": "assistant", "content": proposal.model_dump_json()})
        conversation.append({"role": "user", "content": f"Feedback:\n{feedback}"})

    schema_path = os.path.join(output_dir, "graph_model_schema.json")
    models_path = os.path.join(output_dir, "models.py")

    fmt.echo("\nGenerating outputs...")
    generate_json_schema(proposal, schema_path)
    fmt.success(f"JSON schema written to {schema_path}")
    generate_python_models(proposal, models_path)
    fmt.success(f"Python models written to {models_path}")

    fmt.echo("")
    fmt.echo("You can use the model with:")
    fmt.echo("  # Via JSON schema (API or Python)")
    fmt.echo("  from cognee.shared.graph_model_utils import graph_schema_to_graph_model")
    fmt.echo("  import json")
    fmt.echo(f'  schema = json.load(open("{schema_path}"))')
    fmt.echo("  model = graph_schema_to_graph_model(schema)")
    fmt.echo("  await cognee.cognify(graph_model=model)")
    fmt.echo("")
    fmt.echo("  # Via Python models")
    fmt.echo(f"  from models import {proposal.root_entity_name}")
    fmt.echo(f"  await cognee.cognify(graph_model={proposal.root_entity_name})")
    fmt.echo("")


# ---------------------------------------------------------------------------
# 8. CLI command
# ---------------------------------------------------------------------------


class InferModelCommand(SupportsCliCommand):
    command_string = "infer-model"
    help_string = "Interactively design a graph model for your domain using an LLM"
    docs_url = DEFAULT_DOCS_URL
    description = """
Interactively design a graph model for your domain.

This tool uses an LLM to propose entity types, fields, and relationships
based on your problem description and (optionally) sample data files.
You can refine the proposal through multiple rounds of feedback until
you are satisfied.

Outputs:
- **graph_model_schema.json** — JSON schema usable with the Cognify API
- **models.py** — Production-ready DataPoint subclasses for the Python SDK
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output-dir",
            "-o",
            default=".",
            help="Directory to write output files (default: current directory)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            asyncio.run(_run_interactive(args.output_dir))
        except Exception as exc:
            if isinstance(exc, CliCommandInnerException):
                raise CliCommandException(str(exc), error_code=1) from exc
            raise CliCommandException(f"Error during model inference: {exc}", error_code=1) from exc


if __name__ == "__main__":
    asyncio.run(_run_interactive("."))
