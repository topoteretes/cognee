import pytest


@pytest.mark.deployment
@pytest.mark.asyncio
async def test_basic_harness(running_container, api_client):
    # Start container with SQLite stack (default)
    base_url = running_container("sqlite")

    # Initialize authenticated API client
    client = api_client(base_url)
    await client.register_and_login(email="user@example.com", password="password")

    # 1. Add data to a new dataset
    dataset_name = "test_dataset"
    file_content = "The quick brown fox jumps over the lazy dog."
    files = {"data": ("test.txt", file_content, "text/plain")}
    data = {"datasetName": dataset_name, "run_in_background": "false"}

    print("\nAdding data to dataset...")
    add_response = await client.post("/api/v1/add", files=files, data=data)
    assert add_response.status_code == 200, f"Add failed: {add_response.text}"
    print("Add call succeeded.")

    # 2. Get dataset details to find UUID
    datasets_response = await client.get("/api/v1/datasets")
    assert datasets_response.status_code == 200
    datasets_list = datasets_response.json()

    target_dataset = None
    for d in datasets_list:
        if d["name"] == dataset_name:
            target_dataset = d
            break

    assert target_dataset is not None, "Dataset not found in datasets list"
    dataset_id = target_dataset["id"]
    print(f"Found dataset {dataset_name} with ID: {dataset_id}")

    # 3. Trigger cognify processing
    cognify_payload = {"datasets": [dataset_name], "run_in_background": False}
    print("Triggering cognify processing...")
    cognify_response = await client.post("/api/v1/cognify", json=cognify_payload)
    assert cognify_response.status_code == 200, f"Cognify failed: {cognify_response.text}"
    print("Cognify call succeeded.")

    # 4. Search the processed dataset
    search_payload = {"query": "fox", "search_type": "GRAPH_COMPLETION", "datasets": [dataset_name]}
    print("Searching dataset...")
    search_response = await client.post("/api/v1/search", json=search_payload)
    assert search_response.status_code == 200, f"Search failed: {search_response.text}"

    search_results = search_response.json()
    assert len(search_results) > 0, "No search results returned"

    # Under the mock LLM, search_result should contain the mock API response
    result_text = search_results[0]["search_result"]
    assert result_text is not None
    print("Search call succeeded. Result text:", result_text)

    await client.close()
