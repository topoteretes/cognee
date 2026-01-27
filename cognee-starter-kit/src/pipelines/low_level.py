"""Cognee demo with simplified structure."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from uuid import UUID, uuid4

from pydantic import BaseModel

from cognee import config, prune, search, SearchType, visualize_graph
from cognee.low_level import setup, DataPoint
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.pipelines.operations import run_pipeline
from cognee.modules.users.models import User
from cognee.pipelines import Task
from cognee.tasks.storage import add_data_points
from cognee.modules.users.methods import get_default_user


class Person(DataPoint):
    """Represent a person."""

    name: str
    metadata: dict = {"index_fields": ["name"]}


class Department(DataPoint):
    """Represent a department."""

    name: str
    employees: list[Person]
    metadata: dict = {"index_fields": ["name"]}


class CompanyType(DataPoint):
    """Represent a company type."""

    name: str = "Company"


class Company(DataPoint):
    """Represent a company."""

    name: str
    departments: list[Department]
    is_type: CompanyType
    metadata: dict = {"index_fields": ["name"]}


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
COGNEE_DIR = ROOT / ".cognee_system"
ARTIFACTS_DIR = ROOT / ".artifacts"
GRAPH_HTML = ARTIFACTS_DIR / "graph_visualization.html"
COMPANIES_JSON = DATA_DIR / "companies.json"
PEOPLE_JSON = DATA_DIR / "people.json"


def load_json_file(path: Path) -> Any:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def remove_duplicates_preserve_order(seq: Iterable[Any]) -> list[Any]:
    """Return list with duplicates removed while preserving order."""
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_people_nodes(people: Iterable[Mapping[str, Any]]) -> dict:
    """Build person nodes keyed by name."""
    nodes = {p["name"]: Person(name=p["name"]) for p in people if p.get("name")}
    return nodes


def group_people_by_department(people: Iterable[Mapping[str, Any]]) -> dict:
    """Group person names by department."""
    groups = defaultdict(list)
    for person in people:
        name = person.get("name")
        if not name:
            continue
        dept = person.get("department", "Unknown")
        groups[dept].append(name)
    return groups


def collect_declared_departments(
    groups: Mapping[str, list[str]], companies: Iterable[Mapping[str, Any]]
) -> set:
    """Collect department names referenced anywhere."""
    names = set(groups)
    for company in companies:
        for dept in company.get("departments", []):
            names.add(dept)
    return names


def build_department_nodes(dept_names: Iterable[str]) -> dict:
    """Build department nodes keyed by name."""
    nodes = {name: Department(name=name, employees=[]) for name in dept_names}
    return nodes


def build_company_nodes(companies: Iterable[Mapping[str, Any]], company_type: CompanyType) -> dict:
    """Build company nodes keyed by name."""
    nodes = {
        c["name"]: Company(name=c["name"], departments=[], is_type=company_type)
        for c in companies
        if c.get("name")
    }
    return nodes


def iterate_company_department_pairs(companies: Iterable[Mapping[str, Any]]):
    """Yield (company_name, department_name) pairs."""
    for company in companies:
        comp_name = company.get("name")
        if not comp_name:
            continue
        for dept in company.get("departments", []):
            yield comp_name, dept


def attach_departments_to_companies(
    companies: Iterable[Mapping[str, Any]],
    dept_nodes: Mapping[str, Department],
    company_nodes: Mapping[str, Company],
) -> None:
    """Attach department nodes to companies."""
    for comp_name in company_nodes:
        company_nodes[comp_name].departments = []
    for comp_name, dept_name in iterate_company_department_pairs(companies):
        dept = dept_nodes.get(dept_name)
        company = company_nodes.get(comp_name)
        if not dept or not company:
            continue
        company.departments.append(dept)


def attach_employees_to_departments(
    groups: Mapping[str, list[str]],
    people_nodes: Mapping[str, Person],
    dept_nodes: Mapping[str, Department],
) -> None:
    """Attach employees to departments."""
    for dept in dept_nodes.values():
        dept.employees = []
    for dept_name, names in groups.items():
        unique_names = remove_duplicates_preserve_order(names)
        target = dept_nodes.get(dept_name)
        if not target:
            continue
        employees = [people_nodes[n] for n in unique_names if n in people_nodes]
        target.employees = employees


def build_companies(data: Data) -> list[Company]:
    """Build company nodes from payloads."""
    people = data.people
    companies = data.companies
    people_nodes = build_people_nodes(people)
    groups = group_people_by_department(people)
    dept_names = collect_declared_departments(groups, companies)
    dept_nodes = build_department_nodes(dept_names)
    company_type = CompanyType()
    company_nodes = build_company_nodes(companies, company_type)
    attach_departments_to_companies(companies, dept_nodes, company_nodes)
    attach_employees_to_departments(groups, people_nodes, dept_nodes)
    result = list(company_nodes.values())
    return result


class Data(BaseModel):
    id: UUID
    companies: List[Dict[str, Any]]
    people: List[Dict[str, Any]]


def load_default_payload() -> Data:
    """Load the default payload from data files."""
    companies = load_json_file(COMPANIES_JSON)
    people = load_json_file(PEOPLE_JSON)

    data = Data(
        id=uuid4(),
        companies=companies,
        people=people,
    )

    return data


def ingest_payloads(data: List[Data]) -> list[Company]:
    """Ingest payloads and build company nodes."""
    companies = build_companies(data[0])
    return companies


async def execute_pipeline() -> None:
    """Execute Cognee pipeline."""

    # Configure system paths
    logging.info("Configuring Cognee directories at %s", COGNEE_DIR)
    config.system_root_directory(str(COGNEE_DIR))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Reset state and initialize
    await prune.prune_system(metadata=True)
    await setup()

    # Get user and dataset
    user: User = await get_default_user()  # type: ignore
    dataset = await create_authorized_dataset("demo_dataset", user)
    data = load_default_payload()

    # Build and run pipeline
    tasks = [Task(ingest_payloads), Task(add_data_points)]
    pipeline = run_pipeline(tasks, [data], [dataset.id], user, "demo_pipeline")

    async for status in pipeline:
        logging.info("Pipeline status: %s", status)

    await visualize_graph(str(GRAPH_HTML))

    # Run query against graph
    completion = await search(
        query_text="Who works for GreenFuture Solutions?",
        query_type=SearchType.GRAPH_COMPLETION,
    )
    result = completion
    logging.info("Graph completion result: %s", result)


def configure_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


async def main() -> None:
    """Run main function."""
    configure_logging()
    try:
        await execute_pipeline()
    except Exception:
        logging.exception("Run failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
