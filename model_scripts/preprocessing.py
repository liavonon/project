from os import path, listdir, mkdir
from pydub import AudioSegment
import librosa

# source: path to mp3 file.
# target: path to target wav file.
def convert_to_wav(source, target):
    try:
        sound = AudioSegment.from_file(source)
        sound.export(target, format='wav')
    except:
        print(f'error encountered when trying to convert from m4a to wav source:{source} is probably not formatted correctly...')

# from_dir: origin dir
# to_dir: target dir
def convert_folder_to_wav(from_dir, to_dir):
    if not path.isdir(from_dir):
        raise Exception
    if not path.isdir(to_dir):
        mkdir(to_dir)
    for filename in listdir(from_dir):
        f = path.join(from_dir, filename)
        # checking if it is a file
        if path.isfile(f):
            raw_filename = path.splitext(filename)[0]
            target_path = path.join(to_dir, raw_filename+'.wav')
            convert_to_wav(source=f, target=target_path)

# gets source path, and returns the middle 150 second long audio segment
def get_150sec_mid_audiosegment(source):
    duration = librosa.get_duration(filename=source)
    if duration < 150: 
        print(duration)
        raise Exception('files duration is smaller than 150sec, ignoring')
    left_ms = (duration-150)*1000/2
    right_ms = 150000+left_ms
    audio = AudioSegment.from_file(source)
    return audio[left_ms:right_ms]

# source: path to wav file (thats 150 sec long)
# targets: list of length 30 that contains all the paths of the targets
def cut_song_to_5sec(source, targets):
    if not path.isfile(source):
        raise Exception('source file doesnt exist')
    mid_audio = get_150sec_mid_audiosegment(source)
    
    for p,t in zip(range(150//5), targets):
        left = 5*p*1000
        right = 5*(p + 1)*1000
        part = mid_audio[left:right]
        part.export(t, format='wav')

# cuts entire genre directory containing raw music files (in our case in m4a format) to 5second long cuts and saves them into a new directory 
def cut_genre_to_5sec(from_dir, to_dir):
    if not path.isdir(from_dir):
        raise Exception('from_dir doesnt exist')
    if not path.isdir(to_dir):
        mkdir(to_dir)

    for filename in listdir(from_dir):
        f = path.join(from_dir, filename)
        if path.isfile(f):
            raw_filename = path.splitext(filename)[0]
            target_paths = []
            for i in range(150//5):
                target_paths.append(f'{to_dir}/{raw_filename}_{i}.wav')
            try:
                cut_song_to_5sec(source=f, targets=target_paths)
            except Exception as e:
                print(e)
