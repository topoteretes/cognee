import os
import json
import asyncio
from neo4j import exceptions

from cognee import prune

# from cognee import visualize_graph
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.low_level import setup, DataPoint
from cognee.pipelines import run_tasks, Task
from cognee.tasks.storage import add_data_points


class Products(DataPoint):
    name: str = "Products"


products_aggregator_node = Products()


class Product(DataPoint):
    id: str
    name: str
    type: str
    price: float
    colors: list[str]
    is_type: Products = products_aggregator_node


class Preferences(DataPoint):
    name: str = "Preferences"


preferences_aggregator_node = Preferences()


class Preference(DataPoint):
    id: str
    name: str
    value: str
    is_type: Preferences = preferences_aggregator_node


class Customers(DataPoint):
    name: str = "Customers"


customers_aggregator_node = Customers()


class Customer(DataPoint):
    id: str
    name: str
    has_preference: list[Preference]
    purchased: list[Product]
    liked: list[Product]
    is_type: Customers = customers_aggregator_node


def ingest_files():
    customers_file_path = os.path.join(os.path.dirname(__file__), "customers.json")
    customers = json.loads(open(customers_file_path, "r").read())

    customers_data_points = {}
    products_data_points = {}
    preferences_data_points = {}

    for customer in customers:
        new_customer = Customer(
            id=customer["id"],
            name=customer["name"],
            liked=[],
            purchased=[],
            has_preference=[],
        )
        customers_data_points[customer["name"]] = new_customer

        for product in customer["products"]:
            if product["id"] not in products_data_points:
                products_data_points[product["id"]] = Product(
                    id=product["id"],
                    type=product["type"],
                    name=product["name"],
                    price=product["price"],
                    colors=product["colors"],
                )

            new_product = products_data_points[product["id"]]

            if product["action"] == "purchased":
                new_customer.purchased.append(new_product)
            elif product["action"] == "liked":
                new_customer.liked.append(new_product)

        for preference in customer["preferences"]:
            if preference["id"] not in preferences_data_points:
                preferences_data_points[preference["id"]] = Preference(
                    id=preference["id"],
                    name=preference["name"],
                    value=preference["value"],
                )

            new_preference = preferences_data_points[preference["id"]]
            new_customer.has_preference.append(new_preference)

    return customers_data_points.values()


async def main():
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    await setup()

    pipeline = run_tasks([Task(ingest_files), Task(add_data_points)])

    async for status in pipeline:
        print(status)

    graph_engine = await get_graph_engine()

    products_results = await graph_engine.query(
        """
        // Step 1: Use new customers's preferences from input
        UNWIND $preferences AS pref_input

        // Step 2: Find other customers who have these preferences
        MATCH (other_customer:Customer)-[:has_preference]->(preference:Preference)
          WHERE preference.value = pref_input

        WITH other_customer, count(preference) AS similarity_score

        // Step 3: Limit to the top-N most similar customers
        ORDER BY similarity_score DESC
          LIMIT 5

        // Step 4: Get products that these similar customers have purchased
        MATCH (other_customer)-[:purchased]->(product:Product)

        // Step 5: Rank products based on frequency
        RETURN product, count(*) AS recommendation_score
          ORDER BY recommendation_score DESC
          LIMIT 10
    """,
        {
            "preferences": ["White", "Navy Blue", "Regular Sneakers"],
        },
    )

    print("Top 10 recommended products:")
    for result in products_results:
        print(f"{result['product']['id']}: {result['product']['name']}")

    try:
        await graph_engine.query(
            """
            // Match the customer and their stored shoe size preference
            MATCH (customer:Customer {id: $customer_id})
            OPTIONAL MATCH (customer)-[:has_preference]->(preference:Preference {name: 'ShoeSize'})

            // Assume the new shoe size is passed as a parameter $new_size
            WITH customer, preference, $new_size AS new_size

            // If a stored preference exists and it does not match the new value,
            // raise an error using APOC's utility procedure.
            CALL apoc.util.validate(
              preference IS NOT NULL AND preference.value <> new_size, 
              "Conflicting shoe size preference: existing size is " + preference.value + " and new size is " + new_size, 
              []
            )

            // If no conflict, continue with the update or further processing
            // ...
            RETURN customer
        """,
            {
                "customer_id": "customer_1",
                "new_size": "42",
            },
        )
    except exceptions.ClientError as error:
        print(f"Anomaly detected: {str(error.message)}")

    # # Or use our simple graph preview
    # graph_file_path = str(
    #     os.path.join(os.path.dirname(__file__), ".artifacts/graph_visualization.html")
    # )
    # await visualize_graph(graph_file_path)


if __name__ == "__main__":
    asyncio.run(main())
