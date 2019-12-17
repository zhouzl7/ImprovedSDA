from ImprovedSDA import ImprovedSDA
import LoadData
import Global
import numpy as np

if __name__ == '__main__':
    X1, X2 = LoadData.load_data(Global.FILE_NAME)
    n1 = len(X1)
    n2 = len(X2)
    number = 20
    Y1, X1 = X1[n1-number: n1], X1[: n1-number]
    Y2, X2 = X2[n2-number: n2], X2[: n2-number]
    Y = np.concatenate((Y1, Y2), axis=0)
    isda = ImprovedSDA(X1, X2, Y, minSizeOfSubclass=10)
    predictions = isda.within_predict()
    print(predictions)