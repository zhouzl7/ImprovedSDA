import csv
import numpy as np
import Global


def load_data(fileName):
    with open(fileName, 'r') as csvFile:
        reader = csv.reader(csvFile)
        X1 = []
        X2 = []
        for row in reader:
            v = row
            label = int(v.pop(-1))
            if label > 0:
                X1.append(v)
            else:
                X2.append(v)
    X1 = np.array(X1)  # defective instances
    X2 = np.array(X2)  # defect-free instances
    print(X1.shape)
    print(X2.shape)
    return X1, X2


load_data(Global.FILE_NAME)
