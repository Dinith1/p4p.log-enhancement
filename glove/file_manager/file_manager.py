# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025

import pickle
import os.path
import pandas as pd
from datetime import datetime

MODEL_OBJ_RELATIVE_PATH = './trained_vectors/py_obj/'
PROCESSED_LOG_RELATIVE_PATH = './logs/processed/'


def is_file_exist(path):
    return os.path.isfile(path)


def is_model_obj_exist(name):
    print("Checking for existing" + MODEL_OBJ_RELATIVE_PATH + name + '.pkl model dictionary...')
    return is_file_exist(MODEL_OBJ_RELATIVE_PATH + name + '.pkl')


def save_model(name, obj):
    with open(MODEL_OBJ_RELATIVE_PATH + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_model(name):
    with open(MODEL_OBJ_RELATIVE_PATH + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_csv(obj, log_name, model_name):
    df = pd.DataFrame.from_dict(obj)

    now = datetime.now()
    dt = now.strftime('%d-%m-%YT%H.%M.%S')

    print("Saving csv...")
    df.to_csv(PROCESSED_LOG_RELATIVE_PATH + log_name + "__" + model_name + "__" + dt + ".txt", index=False, sep='\t')
