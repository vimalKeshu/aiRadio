import os 
from collections import namedtuple

spotify_url = "https://api.spotify.com"
user_playlists = "/v1/me/playlists"
playlist_tracks = "/v1/playlists/{playlist_id}/tracks"
audio_analysis = "/v1/audio-features"
top_artists_tracks = "/v1/me/top/{type}"
search = "/v1/search"
add_track_to_playlist = "/v1/playlists/{playlist_id}/tracks"
recommendation = "/v1/recommendations"

os.environ['SPOTIFY_TOKEN'] = ''
access_token = "Bearer " + (os.environ['SPOTIFY_TOKEN'])
spotify_header = {"Content-Type": "application/json",
                  "Accept": "application/json",
                  "Authorization": access_token}
cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo', 'time_signature', 'mode']
        
FP_NAME = "My_Songs"
SP_NAME = "Vimal_PL"

# PID - 4mIhBGKiWY4keoR065OGuy

LIKE_SONG = 0
DISLIKE_SONG = 1