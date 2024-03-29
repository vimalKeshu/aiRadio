import pandas as pd
from collections import namedtuple
from spotify_api import *
from spotify_api import _get
        
Playlist = namedtuple('Playlist', ['id', 'name', 'total'])
Track = namedtuple('Track', ['id', 'name', 'popularity'])
DEFAULT_GENRE = "POP"

def get_playlists_from_json(play_list_json):
    __playlists = {}
    print('play_list_json: ', play_list_json)
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

def get_user_artists_from_json(track_json)-> dict:
    __artists = dict()
    for track in track_json['items']:
        if 'artists' in track:
            for artist in track['artists']:
                __artists.add(artist['id'])
    return __artists

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

def get_playlist_songs(playlist_id, fields):
    __all_tracks = set()

    def __get_next_url(tjson):
        return (tjson['next'] 
                if tjson['next']
                else None)

    def __get_tracks_from_json(tjson):
        __t = set()
        if 'items' in tjson:
            for track in tjson['items']:
                __t.add(track['track']['id'])
        return __t

    __tracks = get_playlist_tracks_by_fields(playlist_id=playlist_id, fields=fields)
    print('__tracks: ',__tracks)
    __all_tracks.update(__get_tracks_from_json(__tracks))
    __next_url = __get_next_url(__tracks)
    print('__next_url: ',__next_url)

    while(__next_url):
        tracks_by_json =  _get(__next_url)
        __all_tracks.update(__get_tracks_from_json(tracks_by_json))
        __next_url = __get_next_url(tracks_by_json)

    return __all_tracks

def search_tracks_by_spotify_recommendation(tracks):
    #print(tracks)
    __recommendate_songs:list = []
    for track in tracks['items']:
        #print(track)
        __artists=[]
        __tracks=[track['track']['id']]
        __genres=set()
        
        for artist in track['track']['artists']:
            __artists.append(artist['id'])

        __artists_details = get_artists_genres(__artists)
        for at in __artists_details['artists']:
            for g in at['genres']:
                if not len(g.split(' ')) > 1:
                    __genres.add(g)

        if not __genres:
            __genres.update([DEFAULT_GENRE])

        #print('all genres: ',__genres)
        # ToDo: choose based on common genres among artists.
        if len(__genres) > 5:
            __genres:set = __genres[:5]

        print('artists',__artists)
        print('tracks',__tracks)
        print('genres: ',__genres)
        cnt=0
        flag=True
        while(cnt < 2 and flag):
            try:
                __track_json = get_recommended_tracks(
                    seed_artists=__artists,
                    seed_genres=list(__genres),
                    seed_tracks=__tracks,
                    limit=5)

                print('recommended', __track_json)
                for t in __track_json['tracks']:
                    #if track['is_playable']:
                    __recommendate_songs.append(t['id'])
                flag=False
            except:
                cnt = cnt + 1
                __genres = [DEFAULT_GENRE]

    return __recommendate_songs
            

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
    + df2["mode"]*100), 3)
    #+ df2["popularity"]), 3)
    ##+ df2["duration_ms"]), 3)    

    df2['distance'] = round(abs(df2['centroid']-df2['location']),3)
    print('Average distance: ', df2['distance'].mean())
    df3 = df2[df2.distance < d]
    df3['new_id'] = 'spotify:track:' + df3['id']
    tracks = __get_tracks_from_dataframe(df3)

    if top and len(tracks) >= top:
        return tracks[:top]
    else:
        return tracks

def find_nearest_neighbour_tracks_per_feature(centroid, df11, tracks, top=None):
    df22 = pd.DataFrame(tracks)
    df2 = pd.merge(df11, df22, on="id")

    df2["distance1"] = df2['danceability']*100 - centroid['danceability']
    df2["distance2"] = df2['energy']*100 - centroid['energy']
    df2["distance3"] = df2['key'] - centroid['energy']
    df2["distance4"] = df2['loudness'] - centroid['loudness']
    df2["distance5"] = df2['speechiness']*100 - centroid['speechiness']
    df2["distance6"] = df2['acousticness']*100 - centroid['acousticness']
    df2["distance7"] = df2['instrumentalness']*100 - centroid['instrumentalness']
    df2["distance8"] = df2['liveness']*100 - centroid['liveness']
    df2["distance9"] = df2['valence']*100 - centroid['valence']
    df2["distance10"] = df2['tempo'] - centroid['tempo']
    df2["distance11"] = df2['time_signature'] - centroid['time_signature']
    df2["distance12"] = df2['mode']*100 - centroid['mode']
    df2["distance13"] = df2['popularity'] - centroid['popularity']


    df2["distance1A"] = df2["distance1"] * df2["distance1"]
    df2["distance2A"] = df2["distance2"] * df2["distance2"]
    df2["distance3A"] = df2["distance3"] * df2["distance3"]
    df2["distance4A"] = df2["distance4"] * df2["distance4"]
    df2["distance5A"] = df2["distance5"] * df2["distance5"]
    df2["distance6A"] = df2["distance6"] * df2["distance6"]
    df2["distance7A"] = df2["distance7"] * df2["distance7"]
    df2["distance8A"] = df2["distance8"] * df2["distance8"]
    df2["distance9A"] = df2["distance9"] * df2["distance9"]
    df2["distance10A"] = df2["distance10"] * df2["distance10"]
    df2["distance11A"] = df2["distance11"] * df2["distance11"]
    df2["distance12A"] = df2["distance12"] * df2["distance12"]
    df2["distance13A"] = df2["distance13"] * df2["distance13"]

    df2['distance'] = ( df2["distance1A"] +
                        df2["distance2A"] +
                        df2["distance3A"] +
                        df2["distance4A"] +
                        df2["distance5A"] +
                        df2["distance6A"] +
                        df2["distance7A"] +
                        df2["distance8A"] +
                        df2["distance9A"] +
                        df2["distance10A"] +
                        df2["distance11A"] +
                        df2["distance12A"] +
                        df2["distance13A"])

    print('Average distance: ', df2['distance'].mean())

    df2['new_id'] = 'spotify:track:' + df2['id']
    tracks = __get_tracks_from_dataframe(df2)

    if top and len(tracks) >= top:
        return tracks[:top]
    else:
        return tracks

def __get_tracks_from_dataframe(df):
    df1 = df.sort_values(by=['distance', 'popularity'], ascending=True)
    records = pd.DataFrame(df1, columns=['new_id', 'name', 'popularity'])
    records.columns = ['id', 'name', 'popularity']
    return list(records.itertuples(index=False, name='Track'))

def publish_tracks(playlist_id, new_tracks, suggested_tracks):
    __tracks = []

    def __publish():
        if not __tracks or len(__tracks) ==0:
            print('There are no tracks suggested.')
        else:
            r = add_tracks_to_playlist(playlist_id, __tracks)
            print(r)

    for t in new_tracks:
        if not t.name in suggested_tracks:
            __tracks.append(t.id)
            suggested_tracks.add(t.name)
        if len(__tracks) == 100:
            __publish()
            __tracks.clear()
    __publish()
    return suggested_tracks

def publish_tracks_by_id(playlist_id, new_tracks):
    __tracks = []

    def __publish():
        if not __tracks or len(__tracks) ==0:
            print('There are no tracks suggested.')
        else:
            r = add_tracks_to_playlist(playlist_id, __tracks)
            print(r)

    for t in new_tracks:
        __tracks.append(t)
        if len(__tracks) == 50:
            __publish()
            __tracks.clear()
    __publish()

