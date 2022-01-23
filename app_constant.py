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

os.environ['SPOTIFY_TOKEN'] = 'BQD75fdzAtcNctP2NyiHObLE8QF_12qS9F7JDSSlSWsGGaK_Dv0SxoVdmYL9APCDwwtZfEXVDc4oYA6ZQ19jGY-aAF6ADj7DtSPPTKV5n4J7k7hqcsGRjrqUMB3Ze-5RZh_5cbbEevMszDZiaPD_J6yoDEjx6GjCpS28I719fGDM18RyPQMYSL2ww46597_fLZol5FYQUdyKNS_QearJEo6YqnvVlS3EYHHVR6qg061p5ufYbKnmkxTyJrow1UKPrl1158PDIcBxo6sS4d4UcwwRUucem_fug5By7-Ar'
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