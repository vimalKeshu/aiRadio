import json
import requests
import os
import math 

import pandas as pd

spotify_url = "https://api.spotify.com"
user_playlists = "/v1/me/playlists"
playlist_tracks = "/v1/playlists/{playlist_id}/tracks"
audio_analysis = "/v1/audio-features"
top_artists_tracks = "/v1/me/top/{type}"
search = "/v1/search"
add_track_to_playlist = "/v1/playlists/{playlist_id}/tracks"

os.environ['SPOTIFY_TOKEN'] = 'BQALAHVQW1O-arREO8xBow98um2pqRTqekhLfl1AQCYpJlsXgYgd6ugcvT3v3bDhK1ySKxZrhDmiviElF-9l583VC-GuadVMO3XQ7KjWBvR6SG3VQybB8hbQ5em5cfoYHR4xDGcRTNY3ypbKs-N4F_Mlp1ubu9yLYnTBOdQEyElF0iCXdKdy_cJQrJahjA_JrBAnKm1UWv7gthLxnsvRZOgVn4OwpSbeTDGG6agGX5F6LKCkRpWPB7PeSK8K06I5ODmzMHwJU4ij8RpEWQ1ZGa10oZPbuEP3hlhs0g'
access_token = "Bearer " + (os.environ['SPOTIFY_TOKEN'])
spotify_header = {"Content-Type": "application/json",
                  "Accept": "application/json",
                  "Authorization": access_token}

cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo', 'time_signature', 'mode']
        
FP_NAME = "My_Songs"
SP_NAME = "Vimal_PL"
_playlists = {}
_tracks = set()
_geners = set()
_suggested_tracks=set()
_likeness_centroid = 0.000

def _get(url):
    return requests.get(url=url,
        headers=spotify_header).json()

def get_user_playlists():
    url = spotify_url + user_playlists
    return requests.get(url=url,
           headers=spotify_header).json()

def get_audio_analysis(tracks):
    url = spotify_url + audio_analysis
    return requests.get(url=url,
            params={"ids": ",".join(tracks)},
            headers=spotify_header).json()

def get_playlist_tracks(playlist_id):
    url = spotify_url + playlist_tracks.format(playlist_id=playlist_id)
    return requests.get(url=url,
            params={"fields": "items(track(id)),next"},
            headers=spotify_header).json()

def get_user_top_charts(type='artists'):
    url = spotify_url + top_artists_tracks.format(type=type)
    return requests.get(url=url,
            params={"time_range": "long_term", "limit":"50"}, #short_term or medium_term
            headers=spotify_header).json()    

def search_tracks_by_gener(gener_type):
    url = spotify_url + search
    return requests.get(url=url,
            params={"q": "\"genre:{gener_type}\" year:2000-2021".format(gener_type=gener_type), "type": "track", "fields": "items(track(id)),next", "limit":"50"}, #short_term or medium_term
            headers=spotify_header).json()

def add_tracks_to_playlist(playlist_id, tracks):
    url = spotify_url + add_track_to_playlist.format(playlist_id=playlist_id)
    r = requests.post(url=url,
           params={"uris": ",".join(tracks)},
           headers=spotify_header) 
    return r

def get_playlists_from_json(play_list_json):
    __playlists = {}
    for pl in play_list_json["items"]:
        __playlists[pl['name']] = pl['id']
    return __playlists

def get_track_ids_from_json(track_json):
    __tracks = set()
    for track in track_json['items']:
        if 'track' in track and 'id' in track['track']:
            __tracks.add(track['track']['id'])
    return __tracks

def get_user_geners_from_json(top_artists_tracks_json):
    __geners = set()
    for item in top_artists_tracks_json['items']:
        if "genres" in item:
            for genre in item['genres']:
                __geners.add(genre)
    return __geners

def get_user_gener_tracks_from_json(tracks_by_gener_json):
    __tracks = set()
    if 'tracks' in tracks_by_gener_json and 'items' in tracks_by_gener_json['tracks']:
        for item in tracks_by_gener_json['tracks']['items']:
            __tracks.add(item['id'])
    return __tracks

def search_tracks_of_gener(gener_type):
    
    def __get_next_url(tjson):
        return (tjson['tracks']['next'] 
                if tjson 
                   and 'tracks' in tjson 
                   and 'next' in tjson['tracks'] 
                else None)

    tracks_by_gener_json = search_tracks_by_gener(gener_type)
    __all_tracks = get_user_gener_tracks_from_json(tracks_by_gener_json)
    __next_url = __get_next_url(tracks_by_gener_json)

    while(__next_url):
       tracks_by_gener_json =  _get(__next_url)
       __all_tracks.update(get_user_gener_tracks_from_json(tracks_by_gener_json))
       __next_url = __get_next_url(tracks_by_gener_json)

    return __all_tracks

def get_centroid_of_audio_features_from_json(audio_features_json):
    df1 = pd.DataFrame(audio_features_json['audio_features'])
    cp = df1.danceability.mean()*100 \
            + df1.energy.mean()*100 \
            + df1.key.mean() \
            + df1.loudness.mean() \
            + df1.speechiness.mean()*100 \
            + df1.acousticness.mean()*100 \
            + df1.instrumentalness.mean()*100 \
            + df1.liveness.mean()*100 \
            + df1.valence.mean()*100 \
            + df1.tempo.mean() \
            + df1.time_signature.mean() \
            + df1["mode"].mean()*100
    return round(cp,3)


def get_audio_analysis_of_all_tracks_as_dataframe(tracks):
    if not tracks:
        print("Don't find any tracks..")
        return

    __tlist = []
    __dataframes = []
    def __get_audio_features():
        tjson = get_audio_analysis(__tlist)
        __dataframes.append(pd.DataFrame(tjson['audio_features']))
    for tid in tracks:
        __tlist.append(tid)
        if len(__tlist) == 100:
            __get_audio_features()
            __tlist.clear()
    __get_audio_features()
    __tlist.clear()

    __df = pd.concat(__dataframes)

    return __df

def find_nearest_neighbour_tracks(centroid, df2, d=1.000):
    df2['centroid'] = round(centroid,3)
    df2["location"] = round((df2['danceability']*100 \
    + df2['energy']*100 \
    + df2['key'] \
    + df2['loudness'] \
    + df2['speechiness']*100 \
    + df2['acousticness']*100 \
    + df2['instrumentalness']*100 \
    + df2['liveness']*100 \
    + df2['valence']*100 \
    + df2['tempo'] \
    + df2['time_signature'] \
    + df2["mode"]*100),3)

    df2['distance'] = round(abs(df2['centroid']-df2['location']),3)
    df3 = df2[df2.distance < d]
    df3['new_id'] = 'spotify:track:' + df3['id']
    print( df3['distance'].tolist())
    return df3['new_id'].tolist()


def publish_tracks(playlist_id, suggested_tracks):
    __tracks = []

    def __publish():
        r = add_tracks_to_playlist(playlist_id, __tracks)
        print(r)

    for t in suggested_tracks:
        if not t in _suggested_tracks:
            __tracks.append(t)
        if len(__tracks) == 100:
            __publish()
            _suggested_tracks.update(_tracks)
            __tracks.clear()
    __publish()

if  '__main__' == __name__:

    _playlists = get_playlists_from_json(get_user_playlists())
    _tracks = get_track_ids_from_json(get_playlist_tracks(_playlists[FP_NAME]))
    _suggested_tracks.update(_tracks)
    _likeness_centroid = get_centroid_of_audio_features_from_json(get_audio_analysis(_tracks))

    print('User likeness centroid: {0}'.format(_likeness_centroid))
    if _likeness_centroid == 0.000:
        print('Error in finding likeness centroid.')
    else:
        _geners = get_user_geners_from_json(get_user_top_charts())
        for gener in _geners:
            search_tracks = search_tracks_of_gener(gener)
            if not search_tracks:
                print('Not able to find any tracks for gener,{0}'.format(gener))
                continue
            #print('search_tracks', search_tracks)
            df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=search_tracks)
            results = find_nearest_neighbour_tracks(centroid=_likeness_centroid, df2=df)
            publish_tracks(playlist_id=_playlists[SP_NAME], suggested_tracks= results)
            print('Suggested the songs for gener, {0}'.format(gener))


    


