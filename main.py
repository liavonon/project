from tkinter import ttk
from model_scripts.data import get_train_val_test_arrays
from model_scripts.training import train_model_and_save_to_path
from model_scripts.eval import evaluate_mode_from_path
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfile
from get_prediction import get_prediction

DEFAULT_PRETRAINED_MODEL_LOCATION = 'saves/model/json/model1.json'
DEFAULT_PRETRAINED_WEIGHTS_LOCATION = 'saves/model/h5/model1.h5'

DEFAULT_TRAINING_MODEL_LOCATION = 'saves/model/json/model.json'
DEFAULT_TRAINING_WEIGHTS_LOCATION = 'saves/model/h5/model.h5'

DEFAULT_DATA_SAVE_LOCATION = 'saves/pickles/data.pickle'

PREDICT_DESCRIPTION = """Choose a mp3, wav or m4a\nfile to recognize genre from"""

SKIP_PREDICT = """If you want to skip to predicting on an existing
model, Press Skip To Predict"""

GO_PREDICT = """When Predicting, it will\nuse this trained model."""

class app:
    def __init__(self, master):
        self.master = master
        self.master.geometry("325x150")
        self.main()

        self.trained_model = False

    def destroy_prev(self):
        for i in self.master.winfo_children():
            i.destroy()

    def main(self):
        self.destroy_prev()
        frame = ttk.Frame(self.master, width=300, height=300)
        frame.pack(pady=20)

        ttk.Label(frame, text='Liav\'s Project').pack(pady=5)

        ttk.Label(frame, text=SKIP_PREDICT).pack(side='top',pady=5)

        self.goto_btn = ttk.Button(frame, text="Go to Preprocessing", command=self.preprocess_window).pack(side='left',pady=5, padx=5)
        
        ttk.Button(frame, text='Skip To Predict', command=self.predict_window).pack(side='left',pady=5, padx=5)
    
    def preprocess_window(self):
        self.destroy_prev()
        frame = ttk.Frame(self.master, width=300, height=300)
        frame.pack(pady=20)

        self.master.title('Preprocessing Data')
        ttk.Label(frame, text=SKIP_PREDICT).pack(pady=5)

        ttk.Button(frame, text='Start', command=lambda:preprocess_data(self)).pack(side='left', pady=5, padx=5)
        
        self.goto_btn = ttk.Button(frame, text="Go to Train", command=self.train_window, state='disabled')
        self.goto_btn.pack(side='left', pady=5, padx=5)

        ttk.Button(frame, text='Skip To Predict', command=self.predict_window).pack(side='left', pady=5, padx=5)

    def train_window(self):
        self.destroy_prev()
        frame = ttk.Frame(self.master, width=300, height=300)
        frame.pack(pady=20)
        
        self.master.title('Train')

        ttk.Label(frame, text=SKIP_PREDICT).pack(pady=5)

        ttk.Button(frame, text='Start',command=lambda:train_model(self)).pack(side='left', pady=5, padx=5)
        
        self.goto_btn = ttk.Button(frame, text="Go to Evaluate", command=self.evaluate_window, state='disabled')
        self.goto_btn.pack(side='left', pady=5, padx=5)

        ttk.Button(frame, text='Skip To Predict', command=self.predict_window).pack(side='left', pady=5, padx=5)

        

    def evaluate_window(self):
        self.destroy_prev()
        frame = ttk.Frame(self.master, width=300, height=300)
        frame.pack(pady=20)
        
        self.master.title('Evaluate')

        ttk.Label(frame, text=GO_PREDICT).pack(side='top',pady=5, padx=5)

        self.label = ttk.Label(frame, text='loss:  accuracy: ')
        self.label.pack(side='top', pady=5, padx=5)

        ttk.Button(frame, text='Start',command=lambda:evaluate(self)).pack(side='left', pady=5, padx=5)

        
        self.goto_btn = ttk.Button(frame, text="Go to Predict", command=self.predict_window).pack(side='left', pady=5, padx=5)

    def predict_window(self):
        self.destroy_prev()
        frame = ttk.Frame(self.master, width=300, height=300)
        frame.pack(pady=20)

        self.master.title('Predict')

        ttk.Label(frame, text=PREDICT_DESCRIPTION).pack(pady=5)

        ttk.Button(frame, text ='Choose File',command = lambda:predict(self)).pack(side='left', pady=5, padx=5)
        
        ttk.Button(frame, text="Exit", command=self.end).pack(side='left', pady=5, padx=5)

        self.label = ttk.Label(frame, text='Prediction: ')
        self.label.pack(side='top', pady=5, padx=5)

    def end(self):
        self.master.destroy()

def preprocess_data(app):
    get_train_val_test_arrays(1000, load_from_pickle=False, pickle_save_path=DEFAULT_DATA_SAVE_LOCATION)
    app.goto_btn['state'] = 'normal'

def train_model(app):
    train_model_and_save_to_path(data_path=DEFAULT_DATA_SAVE_LOCATION, model_save_location=DEFAULT_TRAINING_MODEL_LOCATION, weights_save_location=DEFAULT_TRAINING_WEIGHTS_LOCATION)
    app.goto_btn['state'] = 'normal'
    app.trained_model = True
def evaluate(app):
    loss, accuracy = evaluate_mode_from_path(model_path=DEFAULT_TRAINING_MODEL_LOCATION, weights_path=DEFAULT_TRAINING_WEIGHTS_LOCATION, data_path=DEFAULT_DATA_SAVE_LOCATION)
    loss = str(loss)[:7]
    accuracy = str(accuracy*100)[:7]
    app.label['text'] = f'loss {loss}, accuracy: {accuracy}'

def predict(app):
    file_path = askopenfile(mode='r', filetypes=[('Audio', '*.mp3'), ('Audio', '*.m4a'), ('Audio', '*.wav')])
    if file_path is not None:
        if app.trained_model:   
            model_save_location = DEFAULT_TRAINING_MODEL_LOCATION
            weights_save_location = DEFAULT_TRAINING_WEIGHTS_LOCATION
        else:
            model_save_location = DEFAULT_PRETRAINED_MODEL_LOCATION
            weights_save_location = DEFAULT_PRETRAINED_WEIGHTS_LOCATION
        prediction = get_prediction(file_path.name, model_save_location=model_save_location, weights_save_location=weights_save_location)
        app.label['text'] = f'Prediction:\n{prediction.capitalize()}'
    else:
        raise Exception('not a file')

master = ThemedTk(theme='arc')
master.title('Liav\'s gui')
app(master)
master.mainloop()