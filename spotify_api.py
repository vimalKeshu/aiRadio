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

os.environ['SPOTIFY_TOKEN'] = 'BQC2qKSJThPFrr7qJjjOq9wuo7gjprnVgu0du6SJcfLjjrpRvx3photkWxRiCLreVXwX4DU_GoBKvb5rsKD96CdGoFW4Q87XsS31V4E46_tnXKhw05A9R8X5tpWmnyRA2PbBJY4pB_P7-9qYG2QRgDZt2SsClzI1tpQKKbS48wg_J11gs7tp2duPSao8X2KiLcmLbKDISmb49wq7I26m0BH38E06vG_jKcDYIKGO7Wbp_yxoPoVdvU7AqxA4O-yI88EXQBlTKNkPNz-VeZ67Kge8H6tOZqQFIl3ob31Q'
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
