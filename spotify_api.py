import json
import requests
import os

from app_constant import *

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

def get_playlist_tracks(playlist_id, limit=50):
    url = spotify_url + playlist_tracks.format(playlist_id=playlist_id)
    return requests.get(url=url,
            params={"fields": "items(track(id,name,popularity,duration_ms,artists(id))),next",
                    "limit": limit},
            headers=spotify_header).json()

def get_playlist_tracks_by_fields(playlist_id, fields, limit=50):
    url = spotify_url + playlist_tracks.format(playlist_id=playlist_id)
    return requests.get(url=url,
            params={"fields": fields,
                    "limit": limit},
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

def get_recommended_tracks(seed_artists:list, seed_genres:list, seed_tracks:list, limit=10):
    if not seed_artists or len(seed_artists) > 5:
        raise Exception("Please provide the seed artists upto max 5.")
    if not seed_genres or len(seed_genres) > 5:
        raise Exception("Please provide the seed genres upto max 5.")
    if not seed_tracks or len(seed_tracks) > 5:
        raise Exception("Please provide the seed tracks upto max 5.")
    print('seed_genres',seed_genres)
    url = spotify_url + recommendation
    r = requests.get(url=url,
           params={"seed_artists": ",".join(seed_artists),
           "seed_genres": ",".join(seed_genres),
           "seed_tracks": ",".join(seed_tracks)},
           headers=spotify_header)
    return r.json()

def get_artists_genres(artists_list):
    url = spotify_url + artists
    r = requests.get(url=url,
           params={"ids": ",".join(artists_list)},
           headers=spotify_header)
    return r.json()
