from tensorflow.keras.optimizers import RMSprop
from model_scripts.model import create_model
from model_scripts.data import get_train_val_test_arrays
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
# model_path is the model save path when we finish the training process
MODEL_PATH = 'model_scripts/saves/model/json/model2.json'
WEIGHTS_PATH = 'model_scripts/saves/model/h5/model2.h5'

def train_model_and_save_to_path(data_path, model_save_location='model_scripts/saves/model/json/model2.json', weights_save_location='model_scripts/saves/model/h5/model2.h5'):
    data = get_train_val_test_arrays(pickle_save_path=data_path, load_from_pickle=True)
    model = create_model()
    history = train_model(model, data, epochs=100)
    save_history(history, 'accuracy.png', 'loss.png')
    save_model_and_weights(model, model_save_location, weights_save_location)

    
# model is the model created in create_model function in model.py
# data is a tuple containing all the dataset generators, created in the get_train_test_generators function in data_generators.py
# epochs is the number of epochs in the training process, defaulted to 75.
# returns history, which contains the history of the training process (i.e. loss and acc for each epoch etc.)
def train_model(model : Sequential, data = None, epochs=75):
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    print(y_train, y_val)

    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=2e-5),
                metrics='accuracy')

    history = model.fit(x =X_train, y= y_train, validation_data=(X_val, y_val), epochs=epochs)
    return history

def save_history(history, acc_path, loss_path):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_path)
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_path)

# gets the model which has been trained.
# saves model and weights after the training process to the paths specified in the beginning of the file.
def save_model_and_weights(model, model_save_location, weights_save_location):
    model.save_weights(weights_save_location)
    json = model.to_json()
    with open(model_save_location, 'w') as json_file:
        json_file.write(json)
# at first run, keep load from pickle false,
# at other runs it is suggested to turn it to true (as it would be faster loading the arrays rather then constructing them once again)
# unless it is needed to change num_instances, in which case it would be suggested passing false
if __name__ == '__main__':
    data = get_train_val_test_arrays(num_instances=1500, load_from_pickle=True)
    model = create_model()
    history = train_model(model, data=data)
    save_history(history, 'accuracy.png', 'loss.png')
    save_model_and_weights(model, MODEL_PATH, WEIGHTS_PATH)
    
