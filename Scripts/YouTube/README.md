# RecNet YouTube Data Extractor

Requires Python3.*

**Setup**
1. Generate an OAuth Client Id following steps here: https://developers.google.com/workspace/guides/create-credentials
2. Copy the credentials.json file to your repositories home folder.
3. Follow https://developers.google.com/workspace/guides/configure-oauth-consent to set up an OAuth consent screen for a Google Project that you will create.

**How to fetch data?**
1. Add emails that we will fetch YouTube data for to the test users list on the Google OAuth consent screen.
2. Clone the repository.
3. Run `pip install -r requirements.txt`
4. Run `python fetch_data.py`
5. The above should redirect you to a Google OAuth page asking for YouTube data's readonly access. Accept this using your YT linked email.
6. This will generate 2 files - `<email>_subscriptions.csv` and `<email>_liked_videos.csv` inside the data folder.

**How to run the model?**
- Once data is fetched, simply open a python shell in the home directory.
- Inside the shell run `from prep import *`

This would run the script and generate some recommendations.

**About the model**

You can tweak the following:
```
    MIN_TEXT_LIMIT = 100 # Min text length cutoff to consider a liked video
    TOPIC_COUNT = 15 # Number of topics to classify into when using NMF
    SIMILARITY_THRESHOLD = 0.99999 # The min cosine similarity value to consider
    GROUPING_ATTRIBUTE = 'video_title' # Set to channel_title for channel level classification
```

- We currently use data extracted from YouTube video's transcripts, their tags and description. We do TF-IDF vectorization on top of this and then run NMF (Non-negative matrix factorization) to make topic vectors for each liked video.
- On these topic vectors we check cosine similarity and recommend people with most similar videos as potential friends
