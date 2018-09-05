import requests

base_url = 'http://api.musixmatch.com/ws/1.1/'
api_key = 'apikey=132089eb79411e38f50d1b460db0d233&'

artist_query = base_url + 'artist.search?' + api_key + 'q_artist=prodigy&page_size=10'
r = requests.get(artist_query)
artist_dict = r.json()
artist_list = artist_dict['message']['body']['artist_list']
artist_ids = list()
for i in range(len(artist_list)):
    artist_ids.append(artist_list[i]['artist']['artist_id'])

albums_query = base_url + 'artist.albums.get?' + api_key + 'artist_id=' + str(artist_ids[0]) + '&s_release_date=desc'
r = requests.get(albums_query)
album_dict = r.json()
album_list = album_dict['message']['body']['album_list']
album_ids = list()
for i in range(len(album_list)):
    album_ids.append(album_list[i]['album']['album_id'])

tracksQuery = base_url + 'album.tracks.get?' + api_key + 'album_id=' + str(album_ids[0]) + '&page=1&page_size=2'
r = requests.get(tracksQuery)
tracks_dict = r.json()
tracks_list = tracks_dict['message']['body']['track_list']
track_ids = list()
for i in range(len(tracks_list)):
    track_ids.append(tracks_list[i]['track']['track_id'])

lyricsQuery = base_url + 'track.lyrics.get?' + api_key + 'track_id=15953433'
r = requests.get(lyricsQuery)
lyrics_dict = r.json()
lyrics_body = lyrics_dict['message']['body']['lyrics']['lyrics_body']

print lyrics_body

# TODO: Loop for all lyrics
