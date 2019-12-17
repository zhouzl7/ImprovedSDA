from ImprovedSDA import ImprovedSDA
import LoadData
import Global
import numpy as np

if __name__ == '__main__':
    X1, X2 = LoadData.load_data(Global.SOURCE_FILE_NAME)
    Y1, Y2 = LoadData.load_data(Global.TARGET_FILE_NAME)
    Y = np.concatenate((Y1, Y2), axis=0)
    isda = ImprovedSDA(X1, X2, Y, minSizeOfSubclass=10)
    predictions = isda.within_predict()
    print(predictions)