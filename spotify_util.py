import pandas as pd
from collections import namedtuple
from spotify_api import _get, search_tracks_by_gener, get_audio_analysis, add_tracks_to_playlist


cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo', 'time_signature', 'mode']
        
FP_NAME = "My_Songs"
#FP_NAME = "Sid_PL"
SP_NAME = "Vimal_PL"

Playlist = namedtuple('Playlist', ['id', 'name', 'total'])
Track = namedtuple('Track', ['id', 'name', 'popularity'])

def get_playlists_from_json(play_list_json):
    __playlists = {}
    for pl in play_list_json["items"]:
        __playlists[pl['name']] = pl['id']
    return __playlists

def get_tracks_from_json(track_json):
    __tracks = set()
    for track in track_json['items']:
        if 'track' in track and 'id' in track['track']:
            __tracks.add(Track(track['track']['id'], track['track']['name'], track['track']['popularity']))
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
            __tracks.add(Track(item['id'], item['name'], item['popularity']))
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

def get_centroid_of_audio_features(df11, tracks):
    df22 = pd.DataFrame(tracks)
    df1 = pd.merge(df11, df22, on="id")

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
            + df1["mode"].mean()*100 \
            + df1["popularity"].mean() 
            #+ df1["duration_ms"].mean()
    return round(cp,3)

def get_centroid_of_each_audio_features(df11, tracks):
    df22 = pd.DataFrame(tracks)
    df1 = pd.merge(df11, df22, on="id")

    __centroid = {}

    __centroid['danceability'] = df1.danceability.mean()*100
    __centroid['energy'] = df1.energy.mean()*100
    __centroid['key'] = df1.key.mean()
    __centroid['loudness'] = df1.loudness.mean()
    __centroid['speechiness'] = df1.speechiness.mean()*100
    __centroid['acousticness'] = df1.acousticness.mean()*100
    __centroid['instrumentalness'] = df1.instrumentalness.mean()*100
    __centroid['liveness'] = df1.liveness.mean()*100
    __centroid['valence'] = df1.valence.mean()*100
    __centroid['tempo'] = df1.tempo.mean()
    __centroid['time_signature'] = df1.time_signature.mean()
    __centroid['mode'] = df1["mode"].mean()*100
    __centroid['popularity'] = df1.popularity.mean()

    return __centroid


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

def find_nearest_neighbour_tracks(centroid, df11, tracks, top=None, d=1.000):
    df22 = pd.DataFrame(tracks)
    df2 = pd.merge(df11, df22, on="id")

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
    + df2["mode"]*100 \
    + df2["popularity"]), 3)
    ##+ df2["duration_ms"]), 3)    

    df2['distance'] = round(abs(df2['centroid']-df2['location']),3)
    print('Average distance: ', df2['distance'].mean())
    df3 = df2[df2.distance < d]
    df3['new_id'] = 'spotify:track:' + df3['id']
    df4 = df3.sort_values(by=['distance'], ascending=True)
    #print( df4['distance'].tolist())
    if top and len(df4) >= top:
        return df4['new_id'].tolist()[:top]
    else:
        return df4['new_id'].tolist()

def find_nearest_neighbour_tracks_per_feature(centroid, df11, tracks, top=None):
    df22 = pd.DataFrame(tracks)
    df2 = pd.merge(df11, df22, on="id")

    df2["distance"] = round(( \
        abs(df2['danceability']*100 - centroid['danceability']) \
    +   abs(df2['energy']*100 - centroid['energy']) \
    +   abs(df2['key'] - centroid['energy']) \
    +   abs(df2['loudness'] - centroid['loudness']) \
    +   abs(df2['speechiness']*100 - centroid['speechiness']) \
    +   abs(df2['acousticness']*100 - centroid['acousticness']) \
    +   abs(df2['instrumentalness']*100 - centroid['instrumentalness']) \
    +   abs(df2['liveness']*100 - centroid['liveness']) \
    +   abs(df2['valence']*100 - centroid['valence']) \
    +   abs(df2['tempo'] - centroid['tempo']) \
    +   abs(df2['time_signature'] - centroid['time_signature']) \
    +   abs(df2['mode']*100 - centroid['mode']) \
    +   abs(df2['popularity'] - centroid['popularity'])), 3)

    print('Average distance: ', df2['distance'].mean())

    df2['new_id'] = 'spotify:track:' + df2['id']
    df4 = df2.sort_values(by=['distance'], ascending=True)
    print( df4['distance'].tolist())
    if top and len(df4) >= top:
        return df4['new_id'].tolist()[:top]
    else:
        return df4['new_id'].tolist()

def publish_tracks(playlist_id, new_tracks, suggested_tracks):
    __tracks = []

    def __publish():
        if not __tracks or len(__tracks) ==0:
            print('There are no tracks suggested.')
        else:
            r = add_tracks_to_playlist(playlist_id, __tracks)
            print(r)

    for t in new_tracks:
        if not t in suggested_tracks:
            __tracks.append(t)
        if len(__tracks) == 100:
            __publish()
            suggested_tracks.update([track.id for track in __tracks])
            __tracks.clear()
    __publish()
    return suggested_tracks
