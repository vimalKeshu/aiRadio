import os 
import copy 
import pandas as pd
import numpy as np

from approximator import *
from spotify_util import *
from spotify_api import * 
from predict_tracks import predict

LIKING_SONGS_FILE_NAME = "liking_songs.csv"
NOT_LIKING_SONGS_FILE_NAME = "not_liking_songs.csv"
MODEL_NAME = "songs_recommender"


def __validate_liking_and_not_liking_songs(liking_songs_tracks:set, 
                                            not_liking_songs_tracks:set):
    __not_liking_songs_tracks = set()
    for _tid in not_liking_songs_tracks:
        if _tid in liking_songs_tracks:
            print(f"Found not liking song in liking song list: {_tid}")
        else:
            __not_liking_songs_tracks.add(_tid)

    return (liking_songs_tracks, __not_liking_songs_tracks)    

def collect(liking_songs_path:str, not_liking_songs_path:str) -> None:
    _playlists = get_playlists_from_json(get_user_playlists())
    #print('_playlists:',_playlists)

    _liking_songs_tracks = get_playlist_songs(_playlists[FP_NAME], fields="items(track(id)),next")
    #print('_liking_songs_tracks:',_liking_songs_tracks)

    _not_liking_songs_tracks = get_playlist_songs(_playlists[SP_NAME], fields="items(track(id)),next")
    #print('_not_liking_songs_tracks:',_not_liking_songs_tracks)

    _liking_songs_tracks, _not_liking_songs_tracks = __validate_liking_and_not_liking_songs(
        liking_songs_tracks=_liking_songs_tracks,
        not_liking_songs_tracks=_not_liking_songs_tracks)

    _liked_songs_df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=_liking_songs_tracks)
    _liked_songs_df[cols].to_csv(liking_songs_path, index=False)

    _not_liked_songs_df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=_not_liking_songs_tracks)    
    _not_liked_songs_df[cols].to_csv(not_liking_songs_path, index=False)

def train_model(liking_songs_path:str, 
                not_liking_songs_path:str,
                model_path:str):

    LABEL_COLUMN_NAME = 'label'
    columns = copy.deepcopy(cols)
    print(columns)

    _liked_songs_df = pd.read_csv(liking_songs_path, index_col=columns[0])
    _liked_songs_df[LABEL_COLUMN_NAME] = LIKE_SONG
    print(_liked_songs_df.head)

    _not_liked_songs_df = pd.read_csv(not_liking_songs_path, index_col=columns[0])
    _not_liked_songs_df[LABEL_COLUMN_NAME] = NOT_LIKE_SONG
    print(_not_liked_songs_df.head)

    columns.append(LABEL_COLUMN_NAME)

    songs_df = pd.concat([_liked_songs_df, _not_liked_songs_df])
    songs_df = songs_df[columns[1:]]
    print(songs_df.head)

    songs_np: np.ndarray = songs_df.to_numpy(dtype=np.float64, copy=False)
    np.random.shuffle(songs_np)
    np.random.shuffle(songs_np)
    print(songs_np)

    samples = songs_np[:,:-1]
    print(samples)
    labels = songs_np[:,-1:]
    print(labels)

    approximator: Approximator
    if os.path.exists(model_path):
        print("load the model")
        approximator = Approximator(input_size=len(cols)-1, output_size=1, path=model_path)
    else:
        approximator = Approximator(input_size=len(cols)-1, output_size=1)

    approximator.train(x=samples, y=labels)
    approximator.save(model_path)

if '__main__' == __name__:
    # collect(liking_songs_path=LIKING_SONGS_FILE_NAME, 
    # not_liking_songs_path=NOT_LIKING_SONGS_FILE_NAME)
    train_model(liking_songs_path=LIKING_SONGS_FILE_NAME, 
                not_liking_songs_path=NOT_LIKING_SONGS_FILE_NAME,
                model_path=MODEL_NAME)
    predict(data_path=NOT_LIKING_SONGS_FILE_NAME, model_path=MODEL_NAME)