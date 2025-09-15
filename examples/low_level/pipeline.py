import os
import json
import asyncio
from typing import List, Any
from cognee import prune
from cognee import visualize_graph
from cognee.low_level import setup, DataPoint
from cognee.modules.data.methods import load_or_create_datasets
from cognee.modules.users.methods import get_default_user
from cognee.pipelines import run_tasks, Task
from cognee.tasks.storage import add_data_points


class Person(DataPoint):
    name: str
    # Metadata "index_fields" specifies which DataPoint fields should be embedded for vector search
    metadata: dict = {"index_fields": ["name"]}


class Department(DataPoint):
    name: str
    employees: list[Person]
    # Metadata "index_fields" specifies which DataPoint fields should be embedded for vector search
    metadata: dict = {"index_fields": ["name"]}


class CompanyType(DataPoint):
    name: str = "Company"
    # Metadata "index_fields" specifies which DataPoint fields should be embedded for vector search
    metadata: dict = {"index_fields": ["name"]}


class Company(DataPoint):
    name: str
    departments: list[Department]
    is_type: CompanyType
    # Metadata "index_fields" specifies which DataPoint fields should be embedded for vector search
    metadata: dict = {"index_fields": ["name"]}


def ingest_files(data: List[Any]):
    people_data_points = {}
    departments_data_points = {}
    companies_data_points = {}

    for data_item in data:
        people = data_item["people"]
        companies = data_item["companies"]

        for person in people:
            new_person = Person(name=person["name"])
            people_data_points[person["name"]] = new_person

            if person["department"] not in departments_data_points:
                departments_data_points[person["department"]] = Department(
                    name=person["department"], employees=[new_person]
                )
            else:
                departments_data_points[person["department"]].employees.append(new_person)

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

    return list(companies_data_points.values())


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

    # Run tasks expects a list of data even if it is just one document
    data = [{"companies": companies, "people": people}]

    pipeline = run_tasks(
        [Task(ingest_files), Task(add_data_points)],
        dataset_id=datasets[0].id,
        data=data,
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
