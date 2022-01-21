import os 

from approximator import *
from spotify_util import *
from spotify_api import * 
import pandas as pd
import numpy as np

LIKING_SONGS_FILE_NAME = "liking_songs.csv"
OTHER_SONGS_FILE_NAME = "other_songs.csv"
MODEL_NAME = "songs_recommender"

_playlists = {}
_tracks = set()
_artists = set()
_tracks_object = {}
_geners = set()
_suggested_tracks=set()
_likeness_centroid_per_feature = {}

cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo', 'time_signature', 'mode', 'duration_ms' ]


def predict(path):
    _playlists = get_playlists_from_json(get_user_playlists())
    track_json = get_playlist_tracks(_playlists[FP_NAME])
    _tracks = get_tracks_from_json(track_json=track_json)
    _artists = get_user_artists_from_json(track_json=track_json)
    pass

def train_model(path):
    
    
    
    
    _liked_songs_df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=[t.id for t in _tracks])    
    _liked_songs_df[cols].to_csv(path, index=False)

    df = pd.read_csv(path, index_col=cols[0])
    df = df[cols[1:]]
    print(df.head())
    samples = df.to_numpy(dtype = np.float64, copy = False)
    print(samples)
    print(len(samples))
    labels = np.full((len(samples),2), [0,1], dtype=np.float64)
    print(labels)
    print(len(labels))
    approximator: Approximator
    if os.path.exists(MODEL_NAME):
        print("load the model")
        approximator = Approximator(input_size=len(cols)-1, path=MODEL_NAME)
    else:
        approximator = Approximator(input_size=len(cols)-1, output_size=2)

    approximator.train(x=samples, y=labels)

    print(approximator.predict(x=samples))

    approximator.save(MODEL_NAME)

if '__main__' == __name__:
    #download_liking_songs_features(LIKING_SONGS_FILE_NAME)
    train_model(LIKING_SONGS_FILE_NAME)