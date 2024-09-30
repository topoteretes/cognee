import uuid

import requests
import posthog
import os
from datetime import datetime, timedelta

# Replace with your PostHog Project API Key
POSTHOG_API_KEY = os.getenv('POSTHOG_API_KEY')
POSTHOG_API_HOST =  'https://eu.i.posthog.com'

# Initialize PostHog client
posthog.project_api_key = POSTHOG_API_KEY
posthog.host = POSTHOG_API_HOST

# Read last processed date from file
state_file = 'last_processed_date.txt'
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        last_processed_date = f.read().strip()
        last_processed_date = datetime.strptime(last_processed_date, '%Y-%m-%d')
else:
    # If no state file, start from 2 days ago
    last_processed_date = datetime.utcnow() - timedelta(days=2)

# Calculate the next date to process
next_date = last_processed_date + timedelta(days=1)
today = datetime.utcnow().date()

if next_date.date() >= today:
    print("No new data to process.")
    exit(0)

date_str = next_date.strftime('%Y-%m-%d')

# Fetch download data for the date
package = 'cognee'
url = f'https://pypistats.org/api/packages/{package}/overall'

response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to fetch data: {response.status_code}")
    exit(1)

data = response.json()

# Find the entry for the date we want
downloads = None
for entry in data['data']:
    if entry['date'] == date_str:
        downloads = entry['downloads']
        category = entry.get('category')
        break

if downloads is None:
    print(f"No data available for date {date_str}")
    exit(1)

# Create a unique message_id
message_id = f"cognee_downloads_{date_str}"

distinct_id = str(uuid.uuid4())

# Send an event to PostHog
event_name = 'cognee_lib_downloads_with_mirrors' if category == 'with_mirrors' else 'cognee_lib_downloads_without_mirrors'

if event_name == 'cognee_lib_downloads_without_mirrors':
    posthog.capture(
        distinct_id=str(uuid.uuid4()),
        event=event_name,
        properties={
            'category': category,
            'date': date_str,
            'downloads': downloads,
        }
    )
print(f"Data for {date_str} updated in PostHog successfully. Downloads is {downloads}")

# Update the state file
with open(state_file, 'w') as f:
    f.write(date_str)
