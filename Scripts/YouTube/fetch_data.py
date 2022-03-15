from __future__ import print_function

import os
import json
from pandas import DataFrame as DF

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    'https://www.googleapis.com/auth/youtube.readonly', 
    'https://www.googleapis.com/auth/userinfo.email'
]

api_service_name = "youtube"
api_version = "v3"

def get_user_info(credentials):
    user_info_service = build(
        serviceName='oauth2', version='v2',
        credentials=credentials
    )
    user_info = None
    try:
        user_info = user_info_service.userinfo().get().execute()
    except HttpError as e:
        print('An error occurred: %s', e)

    return user_info


def map_liked_video_data(in_data):
    return {
        "video_title": in_data["snippet"]["title"],
        "video_id": in_data["id"],
        "video_description": in_data["snippet"]["description"],
        "channel_title": in_data["snippet"]["channelTitle"],
        "video_tags": in_data["snippet"].get("tags"),
        "video_duration": in_data["contentDetails"]["duration"],
        "video_stats": in_data["statistics"]
    }

def map_subscribed_data(in_data):
    return {
        "channel_title": in_data["snippet"]["title"],
        "channel_id": in_data["snippet"]["channelId"],
        "channel_description": in_data["snippet"]["description"],
        "total_videos": in_data["contentDetails"]["totalItemCount"],
        "new_videos": in_data["contentDetails"]["newItemCount"],
    }

def get_subscriptions(youtube):
    request = youtube.subscriptions().list(
        part="snippet,contentDetails",
        maxResults=50,
        mine=True
    )
    final_response = []
    while request is not None:
        response = request.execute()
        final_response.extend(response['items'])
        request = youtube.subscriptions().list_next(request, response)

    return list(map(map_subscribed_data, final_response))


def get_liked_videos(youtube):
    # For pagination: https://googleapis.github.io/google-api-python-client/docs/pagination.html
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        maxResults=50,
        myRating="like"
    )

    final_response = []
    while request is not None:
        response = request.execute()
        final_response.extend(response['items'])
        request = youtube.videos().list_next(request, response)

    return list(map(map_liked_video_data, final_response))


def main():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)

    user_email = get_user_info(creds).get("email")

    os.makedirs('data', exist_ok=True)

    try:
        youtube = build(api_service_name, api_version, credentials=creds)

        all_subscriptions = get_subscriptions(youtube)
        print(f"Found {len(all_subscriptions)} subscriptions")
        sub_df = DF(all_subscriptions)
        sub_df.to_csv(f"data/{user_email}_subscriptions.csv")

        all_liked_videos = get_liked_videos(youtube)
        print(f"Found {len(all_liked_videos)} liked videos")
        liked_df = DF(all_liked_videos)
        liked_df.to_csv(f"data/{user_email}_liked_videos.csv")


    except HttpError as err:
        print(err)


if __name__ == '__main__':
    main()