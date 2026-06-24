import pytest


@pytest.mark.deployment
@pytest.mark.asyncio
@pytest.mark.parametrize("db_stack", ["sqlite", "postgres", "neo4j"])
async def test_db_matrix(db_stack, running_container, api_client):
    # Start container with specific db stack
    base_url = running_container(db_stack)

    # Initialize authenticated API client
    client = api_client(base_url)
    await client.register_and_login(email="user@example.com", password="password")

    # 1. Add data to a new dataset
    dataset_name = f"test_{db_stack}_dataset"
    file_content = f"This is testing the {db_stack} database stack in the matrix."
    files = {"data": (f"test_{db_stack}.txt", file_content, "text/plain")}
    data = {"datasetName": dataset_name, "run_in_background": "false"}

    print(f"\n[{db_stack}] Adding data to dataset...")
    add_response = await client.post("/api/v1/add", files=files, data=data)
    assert add_response.status_code == 200, f"[{db_stack}] Add failed: {add_response.text}"
    print(f"[{db_stack}] Add call succeeded.")

    # 2. Get dataset details to find UUID
    datasets_response = await client.get("/api/v1/datasets")
    assert datasets_response.status_code == 200
    datasets_list = datasets_response.json()

    target_dataset = None
    for d in datasets_list:
        if d["name"] == dataset_name:
            target_dataset = d
            break

    assert target_dataset is not None, f"[{db_stack}] Dataset not found in datasets list"
    dataset_id = target_dataset["id"]
    print(f"[{db_stack}] Found dataset {dataset_name} with ID: {dataset_id}")

    # 3. Trigger cognify processing
    cognify_payload = {"datasets": [dataset_name], "run_in_background": False}
    print(f"[{db_stack}] Triggering cognify...")
    cognify_response = await client.post("/api/v1/cognify", json=cognify_payload)
    assert cognify_response.status_code == 200, (
        f"[{db_stack}] Cognify failed: {cognify_response.text}"
    )
    print(f"[{db_stack}] Cognify call succeeded.")

    # 4. Search the processed dataset
    search_payload = {
        "query": "database",
        "search_type": "GRAPH_COMPLETION",
        "datasets": [dataset_name],
    }
    print(f"[{db_stack}] Searching dataset...")
    search_response = await client.post("/api/v1/search", json=search_payload)
    assert search_response.status_code == 200, f"[{db_stack}] Search failed: {search_response.text}"

    search_results = search_response.json()
    assert len(search_results) > 0, f"[{db_stack}] No search results returned"

    result_text = search_results[0]["search_result"]
    assert result_text is not None
    print(f"[{db_stack}] Search call succeeded. Result text:", result_text)

    await client.close()
