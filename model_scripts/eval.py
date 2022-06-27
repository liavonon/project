from tensorflow.keras.models import model_from_json
from model_scripts.data import get_train_val_test_arrays
from tensorflow.keras.optimizers import RMSprop

def get_model_from_path(model_path, weights_path):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    return loaded_model
def evaluate_mode_from_path(model_path='model_scripts/saves/model/json/model2.json', weights_path='model_scripts/saves/model/json/model2.json', data_path='model_scripts/saves/pickles/all_tracks_genres.pickle'):
    model = get_model_from_path(model_path, weights_path)
    data = get_train_val_test_arrays(pickle_save_path=data_path, load_from_pickle=True)[2] # to get test data and not validation or train
    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=5e-5),
                metrics='accuracy')
    return model.evaluate(x=data[0],y=data[1])