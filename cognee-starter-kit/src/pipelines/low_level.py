import os
import uuid
import json
import asyncio
import pathlib
from cognee import config, prune, search, SearchType, visualize_graph
from cognee.low_level import setup, DataPoint
from cognee.pipelines import run_tasks, Task
from cognee.tasks.storage import add_data_points
from cognee.tasks.storage.index_graph_edges import index_graph_edges
from cognee.modules.users.methods import get_default_user


class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Department(DataPoint):
    name: str
    employees: list[Person]
    metadata: dict = {"index_fields": ["name"]}


class CompanyType(DataPoint):
    name: str = "Company"


class Company(DataPoint):
    name: str
    departments: list[Department]
    is_type: CompanyType
    metadata: dict = {"index_fields": ["name"]}


def ingest_files():
    companies_file_path = os.path.join(os.path.dirname(__file__), "../data/companies.json")
    companies = json.loads(open(companies_file_path, "r").read())

    people_file_path = os.path.join(os.path.dirname(__file__), "../data/people.json")
    people = json.loads(open(people_file_path, "r").read())

    people_data_points = {}
    departments_data_points = {}

    for person in people:
        new_person = Person(name=person["name"])
        people_data_points[person["name"]] = new_person

        if person["department"] not in departments_data_points:
            departments_data_points[person["department"]] = Department(
                name=person["department"], employees=[new_person]
            )
        else:
            departments_data_points[person["department"]].employees.append(new_person)

    companies_data_points = {}

    # Create a single CompanyType node, so we connect all companies to it.
    companyType = CompanyType()

    for company in companies:
        new_company = Company(name=company["name"], departments=[], is_type=companyType)
        companies_data_points[company["name"]] = new_company

        for department_name in company["departments"]:
            if department_name not in departments_data_points:
                departments_data_points[department_name] = Department(
                    name=department_name, employees=[]
                )

            new_company.departments.append(departments_data_points[department_name])

    return companies_data_points.values()


async def main():
    cognee_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system")).resolve()
    )
    # Set up the Cognee system directory. Cognee will store system files and databases here.
    config.system_root_directory(cognee_directory_path)

    # Prune system metadata before running, only if we want "fresh" state.
    await prune.prune_system(metadata=True)

    await setup()

    # Generate a random dataset_id
    dataset_id = uuid.uuid4()
    user = await get_default_user()

    pipeline = run_tasks(
        [
            Task(ingest_files),
            Task(add_data_points),
        ],
        dataset_id,
        None,
        user,
        "demo_pipeline",
    )

    async for status in pipeline:
        print(status)

    await index_graph_edges()

    # Or use our simple graph preview
    graph_file_path = str(
        os.path.join(os.path.dirname(__file__), ".artifacts/graph_visualization.html")
    )
    await visualize_graph(graph_file_path)

    # Completion query that uses graph data to form context.
    completion = await search(
        query_text="Who works for GreenFuture Solutions?",
        query_type=SearchType.GRAPH_COMPLETION,
    )
    print("Graph completion result is:")
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
