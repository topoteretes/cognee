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

# Fetch historical download data for the last 180 days
package = 'cognee'
url = f'https://pypistats.org/api/packages/{package}/overall'

response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to fetch data: {response.status_code}")
    exit(1)

data = response.json()

# Exclude today and yesterday
today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)

# Process and send data to PostHog
for entry in data['data']:
    date_str = entry['date']
    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
    downloads = entry['downloads']

    # Skip today and yesterday
    if date_obj >= yesterday:
        continue

    # Create a unique message_id
    message_id = f"cognee_downloads_{date_str}"

    distinct_id = str(uuid.uuid4())

    # Send an event to PostHog
    posthog.capture(
        distinct_id=distinct_id,
        event='cognee_downloads',
        properties={
            'date': date_str,
            'downloads': downloads,
        }
    )

    print(f"Data for {date_str} imported successfully.")

print("Historical data import completed.")
