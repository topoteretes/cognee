import os
import json
import asyncio
from uuid import UUID, uuid4
from pydantic import BaseModel
from typing import Dict, List, Optional

from cognee import prune
from cognee import visualize_graph
from cognee.low_level import setup, DataPoint
from cognee.modules.data.methods import load_or_create_datasets
from cognee.modules.users.methods import get_default_user
from cognee.pipelines import run_tasks, Task
from cognee.tasks.storage import add_data_points


class Person(DataPoint):
    name: str
    # "index_fields": fields to embed for vector search
    # "identity_fields": fields used to generate deterministic IDs (deduplication)
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Department(DataPoint):
    name: str
    employees: list[Person]
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class CompanyType(DataPoint):
    name: str = "Company"
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Company(DataPoint):
    name: str
    departments: list[Department]
    is_type: CompanyType
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Data(BaseModel):
    id: Optional[UUID] = uuid4()
    payload: Dict[str, List]


def ingest_files(data: List[Data]):
    # With identity_fields, DataPoints with the same name automatically get the same UUID.
    # No manual dict-based deduplication needed — just create instances freely.
    all_companies = []

    for data_item in data:
        people = data_item.payload["people"]
        companies = data_item.payload["companies"]

        # Build departments with their employees
        dept_employees: Dict[str, List[Person]] = {}
        for person in people:
            dept_name = person["department"]
            if dept_name not in dept_employees:
                dept_employees[dept_name] = []
            dept_employees[dept_name].append(Person(name=person["name"]))

        departments = {
            name: Department(name=name, employees=employees)
            for name, employees in dept_employees.items()
        }

        # Create a single CompanyType node (deterministic ID via identity_fields)
        company_type = CompanyType()

        for company in companies:
            company_departments = [
                departments.get(dept_name, Department(name=dept_name, employees=[]))
                for dept_name in company["departments"]
            ]
            all_companies.append(
                Company(name=company["name"], departments=company_departments, is_type=company_type)
            )

    return all_companies


async def main():
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    # Create relational database tables
    await setup()

    # If no user is provided use default user
    user = await get_default_user()

    # Create dataset object to keep track of pipeline status
    datasets = await load_or_create_datasets(["test_dataset"], [], user)

    # Prepare data for pipeline
    companies_file_path = os.path.join(os.path.dirname(__file__), "companies.json")
    companies = json.loads(open(companies_file_path, "r").read())
    people_file_path = os.path.join(os.path.dirname(__file__), "people.json")
    people = json.loads(open(people_file_path, "r").read())

    data = Data(payload={"companies": companies, "people": people})

    pipeline = run_tasks(
        [Task(ingest_files), Task(add_data_points)],
        dataset_id=datasets[0].id,
        data=[data],
        incremental_loading=False,
    )

    async for status in pipeline:
        print(status)

    # Or use our simple graph preview
    graph_file_path = str(
        os.path.join(os.path.dirname(__file__), ".artifacts/graph_visualization.html")
    )
    await visualize_graph(graph_file_path)


if __name__ == "__main__":
    asyncio.run(main())
