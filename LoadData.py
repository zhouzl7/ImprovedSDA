import csv
import numpy as np
import Global


def load_data(fileName):
    reader = np.loadtxt(fileName, delimiter=',')
    n = len(reader)
    X1 = []
    X2 = []
    for i in range(n):
        v = reader[i]
        label = v[-1]
        v = np.delete(v, -1)
        if label > 0:
            X1.append(v)
        else:
            X2.append(v)
    X1 = np.array(X1)  # defective instances
    X2 = np.array(X2)  # defect-free instances
    # print(X1.shape)
    # print(X2.shape)
    return X1, X2


# load_data(Global.FILE_NAME)
