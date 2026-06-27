import asyncio
import uuid
from cognee import search, add, cognify
from cognee.modules.users.models import User
from cognee.modules.search.types import SearchType


async def test_isolation():
    # 1. Mock a standard system user configuration
    mock_user = User(id=uuid.uuid4(), email="developer@test.local", tenant_id=uuid.uuid4())

    # 2. Create two distinct datasets
    dataset_starwars = "star_wars_lore"
    dataset_medical = "medical_records"

    # 3. Add distinct data to each dataset
    print("Ingesting content into separate datasets...")
    await add(
        "Eldaa is a legendary Jedi Master from the Old Republic who wields a green lightsaber.",
        dataset_name=dataset_starwars,
        user=mock_user,
    )
    await add(
        "Eldaa is a proprietary Electronic Health Record (EHR) platform used to track clinical metrics.",
        dataset_name=dataset_medical,
        user=mock_user,
    )

    # 4. Build the knowledge graphs
    print("Running graph builds (cognify)...")
    await cognify(dataset_name=dataset_starwars, user=mock_user)
    await cognify(dataset_name=dataset_medical, user=mock_user)

    # 5. Fetch the resolved Dataset database objects to get their UUIDs
    from cognee.modules.data.methods import get_dataset

    sw_db_dataset = await get_dataset(dataset_starwars, user=mock_user)
    med_db_dataset = await get_dataset(dataset_medical, user=mock_user)

    print(f"Star Wars Dataset ID: {sw_db_dataset.id}")
    print(f"Medical Dataset ID: {med_db_dataset.id}")

    # 6. TEST TRIPLE RETRIEVAL: Scope query explicitly to the Star Wars dataset ID
    print("\nExecuting isolated search query against Star Wars dataset...")
    results = await search(
        query_text="What is Eldaa?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[sw_db_dataset.id],  # Strict payload constraint
        user=mock_user,
        verbose=True,
    )

    print("\n--- TEST RESULTS ---")
    print(f"Total datasets found in payload results: {len(results)}")

    for res in results:
        print(f"\nResult Dataset ID: {res.get('dataset_id')}")
        print(f"Result Dataset Name: {res.get('dataset_name')}")
        print(f"Text output context: {res.get('text_result') or res.get('search_result')}")

        # Validation checks
        if res.get("dataset_name") == dataset_medical:
            print("❌ FAILURE: Cross-tenant/Cross-dataset data leaked from Medical cluster data!")
            assert False, "Security Boundary Broken!"

    print("\n✅ SUCCESS: Data isolation verified. Only scoped datasets returned records.")


if __name__ == "__main__":
    asyncio.run(test_isolation())
