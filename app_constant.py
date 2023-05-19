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
artists = "/v1/artists"

os.environ['SPOTIFY_TOKEN'] = 'BQCqnG-yJLinuM15qseuOjj3qrhBUejA4p8eylCyQWfHVK1k5yZFoBWByDc3Z938C8dKPKXF63_zYayEjxUY4zPysg_xgL-TxaEVoresbduBDzBsaw3BxlEUNHGNb_ReAFfY5QvIRRh_dUo_uEs_RAgb5tuFujJWDwJuAwe6u0x8KwGp-8oPVzdoE894ERj4GIh_jdTlwKC0QnPzPxPMqTZ9dp831XSVZ-8s_WAwNTLSEhaOXw7em7ABKZdKs9iRRoiAsLbyk7M6MzA1Xx-Pj96dyrL3N3XnXc4seI4n'
access_token = "Bearer " + (os.environ['SPOTIFY_TOKEN'])
spotify_header = {"Content-Type": "application/json",
                  "Accept": "application/json",
                  "Authorization": access_token}
# cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
#         'liveness','valence', 'tempo', 'time_signature', 'mode']

cols = ['id','danceability','energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo', 'time_signature', 'mode', 'duration_ms' ]
        
FP_NAME = "My_Songs"
SP_NAME = "Vimal_PL"

# PID - 4mIhBGKiWY4keoR065OGuy

LIKE_SONG = 1
NOT_LIKE_SONG = 0