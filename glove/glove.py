import numpy as np
from scipy import spatial
import glove.file_manager.file_manager as fm
import glove.file_manager.log_reader as lr


def process_log(log_path, model_path):
    is_log_exist = fm.is_file_exist(log_path)
    is_model_exist = fm.is_file_exist(model_path)

    if not is_log_exist or not is_model_exist:
        if not is_log_exist:
            print("[ERROR] Could not find log file: " + log_path)
        if not is_model_exist:
            print("[ERROR] Could not find model file: " + model_path)
        return None

    slash = model_path.rfind('/')
    dot = model_path.rfind('.')

    # Find the name of the trained vector model (without its path and extension)
    model_name = model_path[(slash + 1):dot]

    model_obj = {}

    if fm.is_model_obj_exist(model_name):
        print("Using existing model dictionary")
        model_obj = fm.load_model(model_name)
    else:
        print("Creating new model dictionary...")
        model_obj = create_model(model_name)

    log_obj = lr.read_log(log_path)

    print(log_obj)

    return find_closest_embeddings("25", model_obj, 5)


def create_model(name):
    """
    https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    """

    embeddings_dict = {}

    # Assumes that the file is a .txt file
    with open("./trained_vectors/embeddings/" + name + '.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # Save/cache the model to avoid creating embeddings_dict again
    fm.save_model(name, embeddings_dict)

    return embeddings_dict


def find_closest_embeddings(query_word, model, num):
    """
    https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    """

    embedding = model[query_word]

    return sorted(model.keys(), key=lambda word: spatial.distance.euclidean(model[word], embedding))[1:num]
