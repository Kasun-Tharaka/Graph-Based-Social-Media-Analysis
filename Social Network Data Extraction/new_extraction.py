import os
import pandas as pd
from googleapiclient.discovery import build
from textblob import TextBlob

#  Your API key
API_KEY = 'AIzaSyC_pVx7Kc#######3m1sauffEwkPc0'
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 📺 List of video IDs to extract comments from
video_ids = ['MJ1vWb1rGwM', 'p7V4Aa7qEpw', 'p1bfK8ZJgkE', '8vmKtS8W7IQ']  # Replace with actual video IDs

#  CSV file path
CSV_FILE = 'new_krish_youtube_comments.csv'

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'

def extract_comments(video_id, existing_ids):
    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat='plainText'
        ).execute()

        for item in response.get('items', []):
            top_comment = item['snippet']['topLevelComment']
            comment_id = top_comment['id']
            if comment_id in existing_ids:
                continue

            snippet = top_comment['snippet']
            comments.append({
                'comment_id': comment_id,
                'video_id': video_id,
                'parent_id': None,
                'author': snippet.get('authorDisplayName'),
                'author_channel_id': snippet.get('authorChannelId', {}).get('value'),
                'text': snippet.get('textDisplay'),
                'published_at': snippet.get('publishedAt'),
                'updated_at': snippet.get('updatedAt'),
                'like_count': snippet.get('likeCount'),
                'reply_count': item['snippet'].get('totalReplyCount', 0),
                'is_public': True,
                'sentiment': get_sentiment(snippet.get('textDisplay'))
            })

            # Extract replies
            if 'replies' in item:
                for reply in item['replies']['comments']:
                    reply_id = reply['id']
                    if reply_id in existing_ids:
                        continue

                    reply_snippet = reply['snippet']
                    comments.append({
                        'comment_id': reply_id,
                        'video_id': video_id,
                        'parent_id': comment_id,
                        'author': reply_snippet.get('authorDisplayName'),
                        'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value'),
                        'text': reply_snippet.get('textDisplay'),
                        'published_at': reply_snippet.get('publishedAt'),
                        'updated_at': reply_snippet.get('updatedAt'),
                        'like_count': reply_snippet.get('likeCount'),
                        'reply_count': 0,
                        'is_public': True,
                        'sentiment': get_sentiment(reply_snippet.get('textDisplay'))
                    })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

# Load existing data if CSV exists
if os.path.exists(CSV_FILE):
    existing_df = pd.read_csv(CSV_FILE)
    existing_ids = set(existing_df['comment_id'].astype(str))
else:
    existing_df = pd.DataFrame()
    existing_ids = set()

# Extract and append new comments
new_comments = []
for vid in video_ids:
    new_comments.extend(extract_comments(vid, existing_ids))

if new_comments:
    new_df = pd.DataFrame(new_comments)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(CSV_FILE, index=False)
    print(f" Added {len(new_df)} new comments to {CSV_FILE}")
else:
    print(" No new comments found.")
