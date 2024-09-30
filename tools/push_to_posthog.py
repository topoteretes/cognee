# extract_and_push_github_data.py
import uuid

import requests
import os
from posthog import Posthog

# Get environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO = os.getenv('GITHUB_REPOSITORY')
POSTHOG_API_KEY = os.getenv('POSTHOG_API_KEY')  # Your PostHog Project API Key
POSTHOG_HOST = 'https://eu.i.posthog.com' # Default PostHog Cloud

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Initialize PostHog client
posthog = Posthog(
    api_key=POSTHOG_API_KEY,
    host=POSTHOG_HOST
)

posthog.debug = True

def get_repo_info():
    url = f"https://api.github.com/repos/{REPO}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching repo info: {response.status_code}")
        return None

def main():
    repo_info = get_repo_info()

    if repo_info:
        # Prepare data to send to PostHog
        properties = {
            'repo_name': repo_info.get('full_name'),
            'stars': repo_info.get('stargazers_count'),
            'forks': repo_info.get('forks_count'),
            'open_issues': repo_info.get('open_issues_count'),
            'watchers': repo_info.get('subscribers_count'),
            'created_at': repo_info.get('created_at'),
            'updated_at': repo_info.get('updated_at'),
            'pushed_at': repo_info.get('pushed_at'),
            'language': repo_info.get('language'),
            'license': repo_info.get('license').get('name') if repo_info.get('license') else None,
            'topics': repo_info.get('topics')
        }

        print("Repository information: ", properties)

        distinct_id = str(uuid.uuid4())

        # Send event to PostHog
        result = posthog.capture(
            distinct_id=distinct_id,  # You can customize this identifier
            event='cognee_lib_github_repo_stats',
            properties=properties
        )

        print("PostHog response: ", result)

        print("Data sent to PostHog successfully.")
    else:
        print("Failed to retrieve repository information.")

    # Close PostHog client
    posthog.shutdown()

if __name__ == "__main__":
    main()
