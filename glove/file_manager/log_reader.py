import pandas as pd


def read_log(path):
    print("Reading input log...")

    # Assumes that the log is tab separated
    data = pd.read_csv(path, sep='\t')

    return data
