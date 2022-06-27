import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import os
import pickle

# creates melspectrogram from a song path
def get_melspect(file_path):
    if not os.path.isfile(file_path):
        raise Exception(f'file doesnt exist in {file_path} cant create melspect')
    y, sr = librosa.load(file_path, sr=22050)
    melspect = librosa.feature.melspectrogram(y=y)
    melspect = np.array(melspect).reshape(128, 216, 1)
    return melspect

# counts the total tracks, genres labeld, used purely for logging
count = 1
# get a genre directory path containing 5 second long clips of that genre, gets the label of the genre (for example, jazz's label is in our case 0), limit which is the limit of entries from that directory.
# returns tracks, genres arrays. while tracks contain all the spectrograms and genres contain all the labels for them.
def get_tracks_genres_for_dir(dir, label, limit=100):
    if not os.path.isdir(dir):
        raise Exception(f'directory doesnt exist in {dir} cant create tracks, genres')

    tracks, genres = [], []

    for filename, _ in zip(os.listdir(dir), range(limit)):

        if filename == '.gitkeep':
            continue
        fp = os.path.join(dir, filename)
        tracks.append(get_melspect(fp))
        genres.append(label)

        global count
        print(f'done {count}')
        count += 1
    return tracks, genres

# gets label dictionary and limit (which limits the amount of enteries).
# returns all_tracks and genres arrays, while all_tracks contain all the spectrograms and genres contain all the labels for them.
def get_tracks_genres(limit=200, lable_dict={'jazz' : 0, 'trance' : 1}):
    jazz_folder = 'model_scripts/data_post/jazz'
    trance_folder = 'model_scripts/data_post/trance'
    
    jazz_data = get_tracks_genres_for_dir(jazz_folder, label=lable_dict['jazz'], limit=limit//2)
    trance_data = get_tracks_genres_for_dir(trance_folder, label=lable_dict['trance'], limit=limit//2)

    # all tracks will be the X features and genre will be the target y
    genres = jazz_data[1] + trance_data[1]
    all_tracks = jazz_data[0] + trance_data[0]

    return np.array(all_tracks), np.array(genres)

# gets num_instances which is the number of instances from the dataset that we want to train/test/validata on, 
# gets pickle_save_path which is the save path of all_tracks and genres so we wont need to calculate all the spectrograms everytime we want to make changes to the model.
def get_train_val_test_arrays(num_instances=500, pickle_save_path = 'model_scripts/saves/pickles/all_tracks_genres.pickle', load_from_pickle=False):
    all_tracks, genres = None, None
    if load_from_pickle:
        with open(pickle_save_path, 'rb') as f:
            all_tracks, genres = pickle.load(f)
            print(f'loaded from {pickle_save_path}, has {len(genres)} instances')
    else:
        all_tracks, genres = get_tracks_genres(num_instances)
        with open(pickle_save_path, 'wb') as f:
            pickle.dump((all_tracks, genres), f)

    X_train, X_test, y_train, y_test = train_test_split(all_tracks, 
                                                        genres,
                                                        test_size=0.33,
                                                        random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test, 
                                                    y_test,
                                                    test_size=0.5,
                                                    random_state=42)
    return ((X_train, y_train), (X_val, y_val), (X_test, y_test))

if __name__ == '__main__':
    data = get_train_val_test_arrays(num_instances=500, pickle_save_path='pickle_dump', load_from_pickle=True)
    train = data[0]
    print(train[1])