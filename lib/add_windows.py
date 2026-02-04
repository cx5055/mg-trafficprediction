import numpy as np


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    length = len(data)
    end_index = length - window - horizon + 1
    X, Y = [], []
    index = 0

    if single:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window + horizon - 1 : index + window + horizon])
            index += 1
    else:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window : index + window + horizon])
            index += 1

    return np.array(X), np.array(Y)
