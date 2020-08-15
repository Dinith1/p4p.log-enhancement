import numpy as np
from scipy import spatial
import glove.file_manager.file_manager as fm
import glove.file_manager.log_reader as lr
import concurrent.futures
import math


def process_log(log_path, model_path, num_threads):
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

    transformed_log = transform_log(log_obj, model_obj, num_threads)

    slash = log_path.rfind('/')
    dot = log_path.rfind('.')
    log_name = log_path[(slash + 1):dot]

    fm.save_csv(transformed_log, log_name, model_name)

    return transformed_log


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


def transform_log(log, model, num_threads):
    new_log = {'Start time': [], 'End time': [], 'Activity': []}

    num_rows = log["Start time"].size

    num_threads_to_use = 0

    if num_rows < num_threads:
        num_threads_to_use = num_rows
    else:
        num_threads_to_use = num_threads

    to_from = []

    # Determine which part of the input log each thread will work on
    if num_threads_to_use == num_rows:
        for i in range(num_threads_to_use):
            to_from[i] = [i, i + 1]
    else:
        at_least = math.floor(num_rows / num_threads_to_use)
        extra = (num_rows / num_threads_to_use - at_least) * num_threads_to_use

        for i in range(num_threads_to_use):
            to_from[i] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads_to_use) as executor:
        for i in range(num_threads):
            executor.submit(x, to_from[i][0], to_from[i][1], log, model, new_log, i)

    print("Transforming the log using %s threads... This may take a while...", num_threads_to_use)

    return new_log


def x(start_line, end_line, log, model, new_log, thread):
    # For now hardcoded to OrdonezB_Sensors.txt log only\
    for i, row in log.iterrows():
        print(i)

        new_log['Start time'].append(row['Start time'])
        new_log['End time'].append(row['End time'])

        l = row['Location'].lower()
        t = row['Type'].lower()
        p = row['Place'].lower()

        # new_log['Activity'].append(combine_words([l, t, p], model))
        new_log['Activity'].append(find_closest_embeddings(model["king"] - model["male"] + model["female"], model, 5))


def combine_words(words, model):
    # THIS LINE WILL DETERMINE WHAT THE ACTIVITY FOR EACH ROW OF THE LOG WILL BE!
    new_word_embedding = model["in"] + model[words[2]] + model["do"] + model["activity"]

    # for word in words[1:]:
    #     new_word_embedding += model[word]

    return find_closest_embeddings(new_word_embedding, model, 5)[0]


def find_closest_embeddings(embedding, model, num):
    """
    https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    """

    # embedding = model[query_word]

    return sorted(model.keys(), key=lambda word: spatial.distance.euclidean(model[word], embedding))[1:num]
