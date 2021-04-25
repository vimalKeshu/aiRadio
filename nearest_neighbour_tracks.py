from spotify_util import *
from spotify_api import * 

_playlists = {}
_tracks = set()
_tracks_object = {}
_geners = set()
_suggested_tracks=set()
_likeness_centroid = 0.000



if  '__main__' == __name__:

    # get user playlist
    _playlists = get_playlists_from_json(get_user_playlists())
    print(_playlists)

    # get track object
    _tracks = get_tracks_from_json(get_playlist_tracks(_playlists[FP_NAME]))

    # keep list of suggested tracks' id
    for t in _tracks:
        _suggested_tracks.update(t.id)

    # find the centroid
    _likeness_centroid = get_centroid_of_audio_features_from_json(get_audio_analysis([t.id for t in _tracks]), tracks=_tracks)

    print('User likeness centroid: {0}'.format(_likeness_centroid))
    if _likeness_centroid == 0.000:
        print('Error in finding likeness centroid.')
    else:
        top_chart = get_user_top_charts()
        #print('Top chart:', top_chart)
        _geners = get_user_geners_from_json(top_chart)
        for gener in _geners:
            search_tracks = search_tracks_of_gener(gener)
            if not search_tracks:
                print('Not able to find any tracks for gener,{0}'.format(gener))
                continue
            #print('search_tracks', search_tracks)
            df = get_audio_analysis_of_all_tracks_as_dataframe(tracks=[t.id for t in search_tracks])
            results = find_nearest_neighbour_tracks(centroid=_likeness_centroid, df11=df, tracks=search_tracks)
            _suggested_tracks = publish_tracks(playlist_id=_playlists[SP_NAME], new_tracks=results, suggested_tracks=_suggested_tracks)
            print('Suggested the songs for gener, {0}'.format(gener))


    


