import argparse
import os
import re

import requests

parser = argparse.ArgumentParser()  # Creating parser
parser.add_argument("artist_name")
parser.add_argument("page_size")
args = parser.parse_args()
file_object = open('Key.txt')
key = file_object.read()
file_object.close()
base_url = 'http://api.musixmatch.com/ws/1.1/'
api_key = 'apikey=' + key

artist_name = args.artist_name.replace(' ', '%20')
page_size = args.page_size
artist_list = list()
artist_ids = list()
try:
    artist_query = base_url + 'artist.search?' + api_key + 'q_artist=' + artist_name + '&page_size=' + \
                   str(page_size)  # taking the arguments from args
    r = requests.get(artist_query)
    artist_dict = r.json()
    artist_list = artist_dict['message']['body']['artist_list']  # finding the list of artists with the given name
    for i in range(len(artist_list)):
        artist_ids.append(artist_list[i]['artist']['artist_id'])
except:
    print "Couldn't find artist"
    exit()

album_names_ids = dict()
for i in range(len(artist_ids)):
    try:
        albums_query = base_url + 'artist.albums.get?' + api_key + 'artist_id=' + str(artist_ids[i]) \
                       + '&s_release_date=desc'  # Finding the albums of the artists
        r = requests.get(albums_query)
        album_dict = r.json()
        album_list = album_dict['message']['body']['album_list']
        for j in range(len(album_list)):
            value_dict = {'album_name': album_list[j]['album']['album_name'], 'artist_id': artist_ids[i]}
            album_names_ids[album_list[j]['album']['album_id']] = value_dict
    except:
        print "Couldn't find albums"
        exit()

track_ids = dict()
for i in album_names_ids:
    try:
        tracksQuery = base_url + 'album.tracks.get?' + api_key + 'album_id=' + str(i) + \
                      '&page_size=10'  # Finding the tracks of each album
        r = requests.get(tracksQuery)
        tracks_dict = r.json()
        tracks_list = tracks_dict['message']['body']['track_list']
        for j in range(len(tracks_list)):
            track_ids[tracks_list[j]['track']['track_id']] = {'track_name': tracks_list[j]['track']['track_name'],
                                                              'album_id': i}
    except:
        print "Couldn't find tracks"
        exit()

lyrics_list = dict()
error_track_ids = list()
for i in track_ids:
    lyricsQuery = base_url + 'track.lyrics.get?' + api_key + 'track_id=' + str(i)
    r = requests.get(lyricsQuery)  # Finding the lyrics
    lyrics_dict = r.json()
    try:
        lyrics_body = lyrics_dict['message']['body']['lyrics']['lyrics_body']
        lyrics_list[i] = lyrics_body
    except:
        error_track_ids.append(i)

for i in error_track_ids:
    track_ids.pop(i)

home_directory = os.getcwd()
for i in lyrics_list:
    album_id = track_ids[i]['album_id']
    artist_id = album_names_ids[album_id]['artist_id']
    s = '_'
    artist_dir = artist_name.replace('%20', ' ') + s + str(artist_id)
    artist_dir = re.sub(r'[\\/:*?"<>|]', "", artist_dir)
    if artist_dir not in os.listdir(os.getcwd()):
        os.mkdir(artist_dir)
    os.chdir(os.getcwd() + os.sep + artist_dir)
    album_name = album_names_ids[album_id]['album_name']
    album_dir1 = album_name + '_' + str(album_id)
    album_dir2 = re.sub(r'[\\/:*?"<>|]', '', album_dir1)
    album_dir = ''.join(i for i in album_dir2 if ord(i) < 128)
    print album_dir
    if album_dir not in os.listdir(os.getcwd()):
        os.mkdir(album_dir)
    os.chdir(album_dir)
    track_name = track_ids[i]['track_name']
    track_file_name1 = track_name + '_' + str(i)
    track_file_name = re.sub(r'[\\/:*?"<>|]', '', track_file_name1)
    track_file_name += '.txt'
    file_object = open(track_file_name, 'w')
    lyrics = lyrics_list[i][:lyrics_list[i].rfind('******* This Lyrics is NOT for Commercial use *******')]
    ascii_lyrics = ''.join(i for i in lyrics if ord(i) < 128)
    file_object.write(ascii_lyrics)
    os.chdir(home_directory)
