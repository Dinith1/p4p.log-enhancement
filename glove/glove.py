import numpy as np
from scipy import spatial
import glove.file_manager.file_manager as fm


def process_log(log_path, model_path):
    slash = model_path.rfind('/')
    dot = model_path.rfind('.')

    # Find the name of the trained vector model (without its path and extension)
    model_name = model_path[(slash + 1):dot]

    model_obj = {}

    if fm.is_model_exist(model_name):
        model_obj = fm.load_model(model_name)
    else:
        model_obj = create_model(model_name)

    return find_closest_embeddings("dog", model_obj)


def create_model(name):
    """
    https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    """

    embeddings_dict = {}

    # Assumes that the file is a .txt file
    with open("../trained_vectors/embeddings/" + name + ".txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # Save/cache the model to avoid creating embeddings_dict again
    fm.save_model(name, embeddings_dict)

    return embeddings_dict


def find_closest_embeddings(query_word, model):
    """
    https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    """

    embedding = model[query_word]

    return sorted(model.keys(), key=lambda word: spatial.distance.euclidean(model[word], embedding))
