import glob
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.tokenize import RegexpTokenizer
import string
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

MIN_TEXT_LIMIT = 100 # Min text length cutoff to consider a liked video
TOPIC_COUNT = 30 # Number of topics to classify into
SIMILARITY_THRESHOLD = 0.9996 # The min cosine similarity value to consider
GROUPING_ATTRIBUTE = 'video_title' # Set to channel_title for channel level classification

def get_script_from_video_id(video_id):
    """
    Given a YouTube video_id, fetches it's transcript if available
    Returns an empty string otherwise
    """

    script_items = []
    print(f'# Adding for {video_id}')

    try:
        script_items = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except Exception:
        pass

    print(f'# Done for {video_id}')
    return ' '.join([script_item['text'].replace("\n", " ") for script_item in script_items])


def clean_df_script(df):
    df = df.replace(np.nan, '', regex=True)
    df.script = df.script.replace(np.nan, '')
    df.script = df.apply(
        lambda x: x['script'].replace('\n', ' '), axis=1
    )
    df['script'] = df['script'].str.lower()
    df['script'] = df['script'].apply(
        lambda x: ' '.join([word for word in x.split() if len(word)>2])
    )
    return df


def get_all_liked_videos_df(should_clean=False):
    liked_dfs = []
    for fname in glob.glob('data/*_liked_videos.csv'):
        if should_clean:
            liked_dfs.append((fname, clean_df_script(pd.read_csv(fname))))
        else:
            liked_dfs.append((fname, pd.read_csv(fname)))

    return liked_dfs


def get_username_from_fname(fname):
    return '/'.join(fname.split('@')[0].split('/')[1:])


def add_scripts_to_all_liked_videos():
    for (fname, liked_df) in get_all_liked_videos_df():
        print(f'# STARTING TRANSCRIPT EXTRACTION FOR {fname}')

        # Add scripts to liked_videos
        if 'script' not in liked_df.columns:
            liked_df['script'] = liked_df.apply(
                lambda x: get_script_from_video_id(x['video_id']), axis=1
            )

        # Add username to liked_videos
        liked_df['username'] = get_username_from_fname(fname)

        # Clean the df
        liked_df = clean_df_script(liked_df)

        liked_df.to_csv(fname)
        print(f'# DONE TRANSCRIPT EXTRACTION FOR {fname}')

def get_recommendations_for_username(username):
    max_vals = [max(s) for s in new_cosine_sim]
    max_matches_all = np.argsort(max_vals)[::-1]

    max_matches_user = [
        i for i in max_matches_all 
        if (
            i in utoi[username] and 
            max_vals[i] >= SIMILARITY_THRESHOLD
        )
    ][:4]

    ufr_using_channel_map = defaultdict(lambda: defaultdict(list))
    for match_index in max_matches_user:
        matched_index = similarity_indices[match_index][-1]
        if gdf.loc[match_index, 'cluster'] == gdf.loc[matched_index, 'cluster']:
            ufr_using_channel_map[itou[match_index]][itou[matched_index]].append(
                (gdf.loc[match_index, GROUPING_ATTRIBUTE], gdf.loc[matched_index, GROUPING_ATTRIBUTE])
            )

    return ufr_using_channel_map

def print_rec_string(ufr_using_channel_map):
    for (rec_for_user, fdict) in ufr_using_channel_map.items():
        print(f"Recommendation for {rec_for_user}:")
        for (suggested_friend, channel_pairs) in fdict.items():
            print(f"  # Check out {suggested_friend}, you follow similar items like:")
            for p in channel_pairs:
                print(f"    ->Yours: {p[0]} | Theirs: {p[1]}")


"""
Starting main script here
"""
add_scripts_to_all_liked_videos()

dfs = get_all_liked_videos_df(True)
dfs = [d[1] for d in dfs]
df = pd.concat(dfs)
df.video_tags = df.video_tags.replace(np.nan, '')
df.video_description = df.video_description.replace(np.nan, '')

df['video_tags_parsed'] = df['video_tags'].apply(lambda x: pd.eval(x) if x else []).apply(lambda x: ' '.join(x))
df['text'] = df.apply(lambda x: x['script'] + x['video_tags_parsed'] + x['video_description'] + x['video_title'], axis=1)
df['text'] = df['text'].str.replace(
    r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', 
    '', regex=True
)
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' ').translate(str.maketrans('','',string.punctuation)))
df['text'] = df['text'].str.lower()

df = df[(df['text'].str.strip() != '') & (df['text'].str.len() > MIN_TEXT_LIMIT)]
df = df.reset_index()

token = RegexpTokenizer(r'[a-zA-Z]+')
model = TfidfVectorizer(
    analyzer='word', tokenizer = token.tokenize, stop_words='english', 
    min_df = 5, ngram_range = (1,1)
)
v = model.fit_transform(df['text'])

m2 = NMF(n_components=TOPIC_COUNT)    # model selection for topical analysis
dt = m2.fit_transform(v)     # fit model using TFIDF matrix

# create clusters for each liked video
df['cluster'] = dt.argmax(axis=1)

# print the words for each topic
topic_words = []
words = model.get_feature_names()
for r in m2.components_:
    a = sorted([(v,i) for i,v in enumerate(r)],reverse=True)[0:7]        # print 7 most common words
    topic_words.append([words[e[1]] for e in a])

df['cluster2'] = list(dt)
ndf = df.drop(df.columns.difference(['username', GROUPING_ATTRIBUTE, 'cluster2']), axis=1)
gdf = ndf.groupby(['username', GROUPING_ATTRIBUTE]).mean().reset_index()

a = np.stack(gdf['cluster2'].values)

gdf['cluster'] = a.argmax(axis=1)

cosine_sim = cosine_similarity(a, a, dense_output=False)

# Suppress same username similarities
utoidf = gdf.groupby('username', sort=False).apply(lambda x: x.index.tolist())
utoi = utoidf.to_dict()
itou = {
    i: k
    for (k, v) in utoi.items()
    for i in v
}
new_cosine_sim = np.array([
    [item if itou[index] != itou[row_index] else 0 for (index, item) in enumerate(row)]
    for (row_index, row) in enumerate(cosine_sim)
])

similarity_indices = np.argsort(new_cosine_sim, axis=1)

for username in utoi.keys():
    print("---------------------------------------------------------------------")
    print_rec_string(get_recommendations_for_username(username))

print("---------------------------------------------------------------------")
