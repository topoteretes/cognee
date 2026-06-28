import asyncio
import time
import pytest

async def golden_flow(api_client):
    """
    Frozen signature for T1–T13 to build against:
    
    Steps:
    1. Add a tiny known document → POST /api/v1/add
    2. Cognify it → POST /api/v1/cognify
    3. Poll dataset status → GET /api/v1/datasets/{id}/status
    4. Search with GRAPH_COMPLETION + CHUNKS → POST /api/v1/search
    5. Assert the known seeded entity ("Alice") comes back
    
    Returns: assert True if entity found
    """
    
    # Step 1: Add known document
    known_doc = "Alice works at Cognee and manages the AI memory platform."
    
    print("📝 Step 1: Adding known document...")
    files = {"data": ("doc.txt", known_doc.encode("utf-8"), "text/plain")}
    add_response = await api_client.post("/api/v1/add", files=files, data={"datasetName": "test_dataset"})
    assert add_response.status_code == 200, f"Add failed: {add_response.text}"
    
    # Extract dataset_id, falling back to a fixed string if the response structure differs
    resp_json = add_response.json()
    dataset_id = resp_json.get("dataset_id") or resp_json.get("pipeline_run_id") or "test_dataset"
    print(f"✓ Dataset ID: {dataset_id}")
    
    # Step 2: Cognify
    print("🔄 Step 2: Cognifying...")
    cognify_response = await api_client.post("/api/v1/cognify", json={"dataset_ids": [dataset_id]})
    assert cognify_response.status_code == 200, f"Cognify failed: {cognify_response.text}"
    
    # Step 3: Poll status
    print("⏳ Step 3: Polling dataset status...")
    for retry in range(30):
        status_response = await api_client.get(f"/api/v1/datasets/status?dataset={dataset_id}")
        assert status_response.status_code == 200, f"Status poll failed: {status_response.text}"
        
        status_data = status_response.json()
        status_val = str(status_data.get(str(dataset_id), "")).lower()
        
        if "completed" in status_val:
            print("✓ Cognify completed")
            break
        elif "failed" in status_val:
            raise RuntimeError(f"Cognify failed: {status_val}")
        
        await asyncio.sleep(2)
    
    if retry == 29:
        raise TimeoutError(f"Cognify did not complete within 60s")
    
    # Step 4: Search with GRAPH_COMPLETION
    print("🔍 Step 4: Searching...")
    search_response = await api_client.post(
        "/api/v1/search",
        json={
            "query": "Who works at Cognee?",
            "search_type": "GRAPH_COMPLETION"
        }
    )
    assert search_response.status_code == 200, f"Search failed: {search_response.text}"
    
    # Step 5: Assert known entity "Alice" appears (exact string match)
    print("✓ Step 5: Asserting known entity...")
    response_data = search_response.json()
    results = response_data if isinstance(response_data, list) else response_data.get("results", [])
    
    found = any("Alice" in str(r) for r in results)
    
    if not found:
        print(f"❌ Known entity 'Alice' NOT found in results: {results}")
        raise AssertionError("Known entity 'Alice' not found in search results")
    
    print("✓ golden_flow PASSED - Alice found in search results")
    return True
