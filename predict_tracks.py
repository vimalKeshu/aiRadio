import os
from pickle import NONE 

from approximator import *
from spotify_util import *
from spotify_api import * 
import pandas as pd
import numpy as np

OTHER_SONGS_FILE_NAME = "other_songs.csv"
MODEL_NAME = "songs_recommender"

def collect(path):
    _playlists = get_playlists_from_json(get_user_playlists())
    #print(_playlists)
    _track_json = get_playlist_tracks_by_fields(_playlists[FP_NAME], fields="items(track(id,artists(id))),next")
    #print(_track_json)
    _recommendate_songs = search_tracks_by_spotify_recommendation(tracks=_track_json)
    print(_recommendate_songs)
    _suggested_songs_df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=_recommendate_songs)
    _suggested_songs_df[cols].to_csv(path, index=False)

def predict(data_path, model_path):

    if not os.path.exists(model_path):
        raise Exception(f"Model doesn't exist at {model_path}")
    if not os.path.exists(data_path):
        raise Exception(f"Data file doesn't exist at {data_path}")

    df = pd.read_csv(data_path)
    sid:list = df[cols[0]].to_list()
    samples = df[cols[1:]].to_numpy(dtype = np.float64, copy = False)
    #print(samples)
    approximator = Approximator(input_size=len(cols)-1, output_size=1, path=model_path)
    predicted_labels = approximator.predict(x=samples)
    #print(sid)
    print(predicted_labels)
    if len(predicted_labels) != len(sid):
        raise Exception("Input and predicted output size not matching")

    __new_tracks:list=[]
    for i in range(len(sid)):
        if (not np.isnan(predicted_labels[i])) and int(predicted_labels[i]) == LIKE_SONG:
            __new_tracks.append(f"spotify:track:{sid[i]}")

    return __new_tracks

def publish_recommended_songs(new_tracks, playlist_id=None):
    if playlist_id is None:
        _playlists = get_playlists_from_json(get_user_playlists())
        playlist_id = _playlists[SP_NAME]
    publish_tracks_by_id(playlist_id=playlist_id, new_tracks=new_tracks)

if '__main__' == __name__:
    collect(path = OTHER_SONGS_FILE_NAME)
    new_tracks = predict(data_path=OTHER_SONGS_FILE_NAME, model_path=MODEL_NAME)
    publish_recommended_songs(new_tracks)