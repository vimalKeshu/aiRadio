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
recommendation = "/v1/recommendations"

os.environ['SPOTIFY_TOKEN'] = 'BQBMft--wccbUaet4nXEegrsZtG4-9K4R6WPjM4lgHTKUl621u684AnRq5MJWiSCT3JvBw-X1vk_DWV906Q5kP6FCppgLfZrLdwPjL_LGhvEAK9YV8lQ5wHw7ZqErSLf4ikWsdtAUcmLTMApjWT7JTrhLAIdmi7VaPcoTl337fUXjrm64FB5zhtYzDPeUWrKYoYZ-fmzl4mF8skCQNBCefHHt5EH1Ozelq-DKe_Pp0bTRA56u-Px-NMUePTxdAnbgLtPndtp1tJaL3H_EPI94bU_pVIuatthGirlkKoq'
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

def get_recommended_tracks(seed_artists:list, seed_genres:list, seed_tracks:list, limit=10):
    if not seed_artists or len(seed_artists) > 5:
        raise Exception("Please provide the seed artists upto max 5.")
    if not seed_genres or len(seed_genres) > 5:
        raise Exception("Please provide the seed genres upto max 5.")
    if not seed_tracks or len(seed_tracks) > 5:
        raise Exception("Please provide the seed tracks upto max 5.")
    url = spotify_url + recommendation
    r = requests.post(url=url,
           params={"seed_artists": ",".join(seed_artists),
           "seed_genres": ",".join(seed_genres),
           "seed_tracks": ",".join(seed_tracks)},
           headers=spotify_header)