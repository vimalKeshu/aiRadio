import os 

from approximator import *
from spotify_util import *
from spotify_api import * 
import pandas as pd
import numpy as np

LIKING_SONGS_FILE_NAME = "liking_songs.csv"
OTHER_SONGS_FILE_NAME = "other_songs.csv"

_playlists = {}
_tracks = set()
_tracks_object = {}
_geners = set()
_suggested_tracks=set()
_likeness_centroid_per_feature = {}

cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo', 'time_signature', 'mode', 'duration_ms' ]

def download_liking_songs_features(path):
    _playlists = get_playlists_from_json(get_user_playlists())
    _tracks = get_tracks_from_json(get_playlist_tracks(_playlists[FP_NAME]))

    _liked_songs_df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=[t.id for t in _tracks])    
    _liked_songs_df[cols].to_csv(path, index=False)


def train_model(path):
    df = pd.read_csv(path, index_col=cols[0])
    df = df[cols[1:]]
    print(df.head())
    samples = df.to_numpy(dtype = np.float64, copy = False)
    print(samples)
    print(len(samples))
    labels = np.full((len(samples),2), [0,1], dtype=np.float64)
    print(labels)
    print(len(labels))
    approximator = Approximator(input_size=len(cols)-1)
    approximator.train(x=samples, y=labels)

    print(approximator.predict(x=samples))


if '__main__' == __name__:
    #download_liking_songs_features(LIKING_SONGS_FILE_NAME)
    train_model(LIKING_SONGS_FILE_NAME)