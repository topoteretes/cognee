"""Simulate an agent connecting to a Cognee instance.

Usage:
    python tools/simulate_agent.py --type SecFilingAgent --port 8000
    python tools/simulate_agent.py --type SupportBot --port 8000 --data-file /path/to/doc.txt
    python tools/simulate_agent.py --type ResearchAgent --port 8000 --search "What is Cognee?"
"""

import argparse
import secrets
import sys
import uuid

import requests


def main():
    parser = argparse.ArgumentParser(description="Simulate an agent connecting to Cognee")
    parser.add_argument("--type", required=True, help="Agent type name (e.g. SecFilingAgent)")
    parser.add_argument("--port", type=int, default=8000, help="Cognee backend port")
    parser.add_argument("--data-file", help="File to upload as agent memory")
    parser.add_argument("--data-text", help="Text to add as agent memory")
    parser.add_argument("--search", help="Search query to run after adding data")
    parser.add_argument("--cognify", action="store_true", help="Run cognify after adding data")
    parser.add_argument("--dataset", default=None, help="Dataset name (defaults to agent-type)")
    args = parser.parse_args()

    base = f"http://localhost:{args.port}"
    short_id = uuid.uuid4().hex[:6]
    email = f"{args.type}-{short_id}@cognee.agent"
    password = secrets.token_hex(16)
    dataset_name = args.dataset or args.type.lower().replace(" ", "-")

    print(f"[Agent] Type: {args.type}")
    print(f"[Agent] ID:   {short_id}")
    print(f"[Agent] Email: {email}")
    print()

    # 1. Register
    print("[1/6] Registering agent...")
    r = requests.post(
        f"{base}/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "is_verified": True,
        },
    )
    if r.status_code == 201 or r.status_code == 200:
        user_data = r.json()
        print(f"  Registered: {user_data.get('id', 'ok')}")
    elif r.status_code == 400 and "REGISTER_USER_ALREADY_EXISTS" in r.text:
        print("  Already registered.")
    else:
        print(f"  Failed: {r.status_code} {r.text}")
        sys.exit(1)

    # 2. Login
    print("[2/6] Logging in...")
    r = requests.post(
        f"{base}/api/v1/auth/login",
        data={
            "username": email,
            "password": password,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if r.status_code != 200:
        print(f"  Login failed: {r.status_code} {r.text}")
        sys.exit(1)
    token = r.json()["access_token"]
    print(f"  Token: {token[:20]}...")

    auth = {"Authorization": f"Bearer {token}"}

    # 3. Create API key
    print("[3/6] Creating API key...")
    r = requests.post(
        f"{base}/api/v1/auth/api-keys",
        headers=auth,
        json={
            "name": f"{args.type}-{short_id}",
        },
    )
    if r.status_code in (200, 201):
        key_data = r.json()
        api_key = key_data.get("key") or key_data.get("api_key", "")
        print(f"  API Key: {api_key[:16]}...")
    else:
        print(f"  Failed: {r.status_code} {r.text}")
        # Try to continue with bearer token
        api_key = None

    api_auth = {"X-Api-Key": api_key} if api_key else auth

    # 4. Add data
    if args.data_file or args.data_text:
        print(f"[4/6] Adding data to dataset '{dataset_name}'...")
        if args.data_file:
            with open(args.data_file, "rb") as f:
                r = requests.post(
                    f"{base}/api/v1/add",
                    headers=api_auth,
                    files={
                        "data": (args.data_file.split("/")[-1], f),
                    },
                    data={"datasetName": dataset_name},
                )
        else:
            r = requests.post(
                f"{base}/api/v1/add",
                headers={**api_auth, "Content-Type": "application/json"},
                json={
                    "textData": [args.data_text],
                    "datasetName": dataset_name,
                },
            )
        if r.status_code == 200:
            info = r.json()
            print(f"  Added: dataset_id={info.get('dataset_id', 'ok')}")
        else:
            print(f"  Failed: {r.status_code} {r.text[:200]}")
    else:
        print("[4/6] Skipping data upload (no --data-file or --data-text)")

    # 5. Cognify
    if args.cognify:
        print(f"[5/6] Cognifying dataset '{dataset_name}'...")
        r = requests.post(
            f"{base}/api/v1/cognify",
            headers={**api_auth, "Content-Type": "application/json"},
            json={
                "datasets": [dataset_name],
                "runInBackground": False,
            },
        )
        if r.status_code == 200:
            print("  Cognify complete.")
        else:
            print(f"  Failed: {r.status_code} {r.text[:200]}")
    else:
        print("[5/6] Skipping cognify (use --cognify)")

    # 6. Search
    if args.search:
        print(f'[6/6] Searching: "{args.search}"...')
        r = requests.post(
            f"{base}/api/v1/search",
            headers={**api_auth, "Content-Type": "application/json"},
            json={
                "query": args.search,
                "searchType": "GRAPH_COMPLETION",
                "datasets": [dataset_name],
            },
        )
        if r.status_code == 200:
            results = r.json()
            for item in results if isinstance(results, list) else []:
                for text in item.get("search_result", []):
                    print(f"  → {text}")
        else:
            print(f"  Failed: {r.status_code} {r.text[:200]}")
    else:
        print("[6/6] Skipping search (use --search 'query')")

    print()
    print(f"Agent '{args.type}-{short_id}' connected successfully.")
    print("Check the UI at http://localhost:3000/connections")


if __name__ == "__main__":
    main()
