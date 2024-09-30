import tweepy
import requests
import json
from datetime import datetime

# Twitter API credentials from GitHub Secrets
API_KEY = '${{ secrets.TWITTER_API_KEY }}'
API_SECRET = '${{ secrets.TWITTER_API_SECRET }}'
ACCESS_TOKEN = '${{ secrets.TWITTER_ACCESS_TOKEN }}'
ACCESS_SECRET = '${{ secrets.TWITTER_ACCESS_SECRET }}'
USERNAME = '${{ secrets.TWITTER_USERNAME }}'
SEGMENT_WRITE_KEY = '${{ secrets.SEGMENT_WRITE_KEY }}'

# Initialize Tweepy API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
twitter_api = tweepy.API(auth)

# Segment endpoint
SEGMENT_ENDPOINT = 'https://api.segment.io/v1/track'


def get_follower_count(username):
    try:
        user = twitter_api.get_user(screen_name=username)
        return user.followers_count
    except tweepy.TweepError as e:
        print(f'Error fetching follower count: {e}')
        return None


def send_data_to_segment(username, follower_count):
    current_time = datetime.now().isoformat()

    data = {
        'userId': username,
        'event': 'Follower Count Update',
        'properties': {
            'username': username,
            'follower_count': follower_count,
            'timestamp': current_time
        },
        'timestamp': current_time
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {SEGMENT_WRITE_KEY.encode("utf-8").decode("utf-8")}'
    }

    try:
        response = requests.post(SEGMENT_ENDPOINT, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            print(f'Successfully sent data to Segment for {username}')
        else:
            print(f'Failed to send data to Segment. Status code: {response.status_code}, Response: {response.text}')
    except requests.exceptions.RequestException as e:
        print(f'Error sending data to Segment: {e}')


follower_count = get_follower_count(USERNAME)
if follower_count is not None:
    send_data_to_segment(USERNAME, follower_count)
else:
    print('Failed to retrieve follower count.')
