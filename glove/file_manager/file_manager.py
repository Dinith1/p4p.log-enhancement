# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025

import pickle
import os.path

currDir = os.path.dirname(os.path.realpath(__file__))
trainedVectorsFolder = os.path.join(currDir, '../trained_vectors/')

# Needed for os.path.isfile() function
TRAINED_VECTORS_EMBEDDINGS_FOLDER = trainedVectorsFolder + 'embeddings/'
TRAINED_VECTORS_OBJ_FOLDER = trainedVectorsFolder + 'py_obj/'

MODELS_RELATIVE_PATH = '../../trained_vectors/py_obj/'


def is_model_exist(name):
    return os.path.isfile(TRAINED_VECTORS_OBJ_FOLDER + name + '.pkl')


def save_model(name, obj):
    with open(MODELS_RELATIVE_PATH + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_model(name):
    with open(MODELS_RELATIVE_PATH + name + '.pkl', 'rb') as f:
        return pickle.load(f)
