from tensorflow.keras.models import model_from_json
import librosa
import os
from pydub import AudioSegment

MODEL_PATH = 'model_scripts/saves/model/json/model.json'
WEIGHTS_PATH = 'model_scripts/saves/model/h5/model.h5'

# gets model_path and weights_path in order to load the model from them.
# returns the loaded model.
def _load_model(model_path='models/model.json', weights_path='models/weights.h5'):
    if not os.path.isfile(model_path):
        raise Exception(f'model is not in {model_path}')
    if not os.path.isfile(weights_path):
        raise Exception(f'weights is not in {weights_path}')
    with open(model_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

model = None
# gets the prediction of the genre from an audio file.
def get_prediction(audio_file_path, model_save_location=MODEL_PATH, weights_save_location=WEIGHTS_PATH):
    global model
    if model is None:
        if not (os.path.isfile(model_save_location) and os.path.isfile(weights_save_location)):
            weights_save_location = WEIGHTS_PATH
            model_save_location = MODEL_PATH
        # initiallizing the model for further use.
        model = _load_model(model_save_location, weights_save_location)
    targets = [f'temp/{i}.wav' for i in range(6)]
    _cut_song_to_5sec(audio_file_path, targets)
    prediction = _predict(model, targets)
    return prediction

# jazz = 0, trance = 1
labels = ['jazz', 'trance']
# gets model and wav_paths (which are the paths of the 5 second cuts (usually 6 of them are created although it is changeable))
# returns the most likely genre for all the cuts paths it gets (bt calculating the average of the predictions)
def _predict(model, wav_paths):
    pred_vals = []
    for p in wav_paths:
        pred_vals.append(_predict_single(model, p))
    final_pred = round(sum(pred_vals) / len(pred_vals))
    return labels[final_pred]

# gets the path of 5 sec long wav file
# returns the melspectrogram of a wav file
def _get_melspect(song_path):
    if not os.path.isfile(song_path):
        raise Exception(f'song doesnt exist in {song_path} cant create melspect')
    try:
        y, sr = librosa.load(song_path, sr=22050)
        melspect = librosa.feature.melspectrogram(y=y)
        return melspect
    except Exception as e:
        raise Exception('file not formatted correctly/corrupt')

# gets model and path of a 5 sec wav file and uses the model to generate a prediction for the file
# returns prediction of a single wav file's genre
def _predict_single(model, wav_path):
    prediction = model.predict(_get_melspect(wav_path).reshape(1,128,216,1))
    pred_val = prediction[0, 0] # prediction looks like [[pred_val]]
    return pred_val

# source: path to wav file (thats 30 sec long)
# targets: list of length 5 that contains all the paths of the targets
def _cut_song_to_5sec(source, targets):
    if not os.path.isfile(source):
        raise Exception('source file doesnt exist')
    mid_audio = _get_30sec_mid_audiosegment(source)
    
    for p,t in zip(range(30//5), targets):
        left = 5*p*1000
        right = 5*(p + 1)*1000
        part = mid_audio[left:right]
        part.export(t, format='wav')

# gets source path, and returns the middle 150 second long audio segment
def _get_30sec_mid_audiosegment(source):
    duration = librosa.get_duration(filename=source)
    if duration < 30: 
        print(duration)
        raise Exception('files duration is smaller than 150sec, ignoring')
    left_ms = (duration-150)*1000/2
    right_ms = 30000+left_ms
    audio = AudioSegment.from_file(source)
    return audio[left_ms:right_ms]

