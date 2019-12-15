import LoadData
import Global
import numpy as np


def get_label_of_y(X1, X2, h1, h2, y):
    subX1, subX2 = NNC(X1, X2, h1, h2)
    # TODO: step 3 - 6

    return Global.POSITIVE


# done!
def get_skewed_F_measure(X1, X2, h1, h2, alpha):
    TP = 0  # the number of defective instances that are predicted as defective (true judgment)
    TN = 0  # the number of defect-free instances that are predicted as defect-free (true judgment)
    FP = 0  # the number of defect-free instances that are predicted as defective (misjudgement)
    FN = 0  # the number of defective instances that are predicted as defect-free (misjudgement)

    # predicted defective instances
    n1 = len(X1)
    for i in range(0, n1):
        # y = X1.pop(i)
        y = X1[i]
        X1 = np.delete(X1, i, axis=0)
        if get_label_of_y(X1, X2, h1, h2, y):
            TP = TP + 1
        else:
            FN = FN + 1
        # X1.insert(i, y)
        X1 = np.insert(X1, i, y, axis=0)

    # predicted defect-free instances
    n2 = len(X2)
    for i in range(0, n2):
        # y = X2.pop(i)
        y = X2[i]
        X2 = np.delete(X2, i, axis=0)
        if get_label_of_y(X1, X2, h1, h2, y):
            FP = FP + 1
        else:
            TN = TN + 1
        # X2.insert(i, y)
        X2 = np.insert(X2, i, y, axis=0)

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Pf = FP / (FP + TN)  # false positive rate
    TNR = 1 - Pf  # true negative rate
    skewedFMeasure = (1 + alpha) * Precision * Recall / (alpha * Precision + Recall)
    return skewedFMeasure


# done!
def get_H1_H2_for_I_SDA(X1, X2):
    n1 = len(X1)
    n2 = len(X2)
    H1 = 0
    H2 = 0
    skewedFMeasure = -1
    h1 = 2
    while h1 * 2 <= n1:
        if n1 % h1 != 0:
            h1 = h1 + 1
        else:
            h2 = round(n2 / (n1 / h1))
            sf = get_skewed_F_measure(X1, X2, h1, h2, 4)
            if skewedFMeasure < sf:
                skewedFMeasure = sf
                H1 = h1
                H2 = h2
            h1 = h1 + 1
    return H1, H2


# done!
def sort_for_nnc(X):
    n, m = X.shape
    sortedX = np.zeros(shape=(n, m))
    euclideanDistance = np.zeros(shape=(n, n))
    maxDistance = -1
    s = 0
    b = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            euclideanDistance[i][j] = np.linalg.norm(X[i] - X[j])
            euclideanDistance[j][i] = euclideanDistance[i][j]
            if maxDistance < euclideanDistance[i][j]:
                maxDistance = euclideanDistance[i][j]
                s = i
                b = j
    sortedX[0] = X[s]
    sortedX[n-1] = X[b]
    euclideanDistance[s][b] = float('inf')
    euclideanDistance[b][s] = float('inf')
    for g in range(0, int((n-1)/2)):
        minDistance = float('inf')
        m = 0
        for j in range(0, n):
            if euclideanDistance[s][j] < minDistance and j != s:
                minDistance = euclideanDistance[s][j]
                m = j
        sortedX[g+1] = X[m]
        euclideanDistance[s][m] = float('inf')
        euclideanDistance[b][m] = float('inf')

        if g+1 != n-g-2:
            minDistance = float('inf')
            k = 0
            for j in range(0, n):
                if euclideanDistance[b][j] < minDistance and j != b:
                    minDistance = euclideanDistance[b][j]
                    k = j
            sortedX[n-g-2] = X[k]
            euclideanDistance[s][k] = float('inf')
            euclideanDistance[b][k] = float('inf')
    return sortedX


# done!
# def divide_to_subclass(X, H):
#     X = X.tolist()
#     n = len(X)
#     subX = []
#     for i in range(H):
#         one_subX = X[int(i*n/H): int((i+1)*n/H)]
#         subX.append(one_subX)
#     return np.array(subX)


# done!
def NNC(X1, X2, H1, H2):
    sortedX1 = sort_for_nnc(X1)
    sortedX2 = sort_for_nnc(X2)
    subX1 = np.array_split(sortedX1, H1)
    subX2 = np.array_split(sortedX2, H2)
    return subX1, subX2


# done!
# subX1 : list of np.array
def get_sumB(subX1, subX2, n1, n2):
    n = n1 + n2  # the number of all samples
    H1 = len(subX1)
    H2 = len(subX2)
    sum_B = 0
    for i in range(H1):
        p_1i = len(subX1[i]) / n
        u_1i = np.mean(subX1[i], axis=0)
        u_1i = np.array([u_1i.tolist()])
        for j in range(H2):
            p_2j = len(subX2[j]) / n
            u_2j = np.mean(subX2[j], axis=0)
            u_2j = np.array([u_2j.tolist()])
            gap = u_1i-u_2j
            sum_B = sum_B + (p_1i * p_2j * np.dot(gap.T, gap))
    return sum_B


# done!
def get_sumX(X1, X2, n1, n2):
    u1 = np.mean(X1, axis=0)
    u2 = np.mean(X2, axis=0)
    u = (u1 + u2) / 2
    u = np.array([u.tolist()])
    sum_X = 0
    for i in range(n1):
        x = np.array([X1[i].tolist()])
        gap = x - u
        sum_X = sum_X + np.dot(gap, gap.T)
    for i in range(n2):
        x = np.array([X2[i].tolist()])
        gap = x - u
        sum_X = sum_X + np.dot(gap.T, gap)
    return sum_X


def improved_SDA(X1, X2, y):
    n1 = len(X1)
    n2 = len(X2)
    H1, H2 = get_H1_H2_for_I_SDA(X1, X2)
    subX1, subX2 = NNC(X1, X2, H1, H2)
    sum_B = get_sumB(subX1, subX2, n1, n2)
    sum_X = get_sumX(X1, X2, n1, n2)
    # TODO: step 4 - 6

    return Global.POSITIVE


a = np.arange(1440).reshape((72, 20))
print(a.shape)
b = np.arange(600).reshape((30, 20))
print(b.shape)
subA, subB = NNC(a, b, 7, 3)
sum_B = get_sumB(subA, subB, 72, 30)
print(sum_B.shape)
sum_X = get_sumX(a, b, 72, 30)
print(sum_X.shape)
