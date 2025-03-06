import os
import json
import asyncio
from cognee import prune
from cognee import visualize_graph
from cognee.low_level import setup, DataPoint
from cognee.pipelines import run_tasks, Task
from cognee.tasks.storage import add_data_points
from cognee.shared.utils import render_graph


class Person(DataPoint):
    name: str


class Department(DataPoint):
    name: str
    employees: list[Person]


class CompanyType(DataPoint):
    name: str = "Company"


class Company(DataPoint):
    name: str
    departments: list[Department]
    is_type: CompanyType


def ingest_files():
    companies_file_path = os.path.join(os.path.dirname(__file__), "companies.json")
    companies = json.loads(open(companies_file_path, "r").read())

    people_file_path = os.path.join(os.path.dirname(__file__), "people.json")
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
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    await setup()

    pipeline = run_tasks([Task(ingest_files), Task(add_data_points)])

    async for status in pipeline:
        print(status)

    # Get a graphistry url (Register for a free account at https://www.graphistry.com)
    await render_graph()

    # Or use our simple graph preview
    graph_file_path = str(
        os.path.join(os.path.dirname(__file__), ".artifacts/graph_visualization.html")
    )
    await visualize_graph(graph_file_path)


if __name__ == "__main__":
    asyncio.run(main())
