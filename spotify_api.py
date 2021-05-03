import json
import requests
import os

spotify_url = "https://api.spotify.com"
user_playlists = "/v1/me/playlists"
playlist_tracks = "/v1/playlists/{playlist_id}/tracks"
audio_analysis = "/v1/audio-features"
top_artists_tracks = "/v1/me/top/{type}"
search = "/v1/search"
add_track_to_playlist = "/v1/playlists/{playlist_id}/tracks"
recommendation_url = "/v1/recommendations"

os.environ['SPOTIFY_TOKEN'] = 'BQC8E3Ae5Lgeh9j2nW2rwadLzdtSBBamWKQNxxoAGYGogV-JBERRBz_Gk8J5rmmB59-5rwOigo7wgZNYbnDpOyi77x3US9b9H3fxcPbyZJ9ehc3o8c0AELN9KCHjOngBfVdOUcrOiDfo7HNLJPBpR1OJ_acyAu7j6iJDEDx0nxn9n7IWgmo1rufkNPpySN4po6Qti8ahb0ULz3LsNqR9s1B7GFpEbrMFe5GH3sZc9PdSN2S6h0oxyOFwKiAt-wfQnD_zFA1COeA49WV1sAG-o7svclyeQC_LIhvraRR9'

access_token = "Bearer " + (os.environ['SPOTIFY_TOKEN'])
spotify_header = {"Content-Type": "application/json",
                  "Accept": "application/json",
                  "Authorization": access_token}

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
            params={"fields": "items(track(id,name,popularity,duration_ms)),next"},
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
