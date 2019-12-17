import numpy as np

a = np.arange(24).reshape(8, 3)
print(a)
b = np.arange(12).reshape(4, 3)
print(b)
c = np.hstack((a.T, b.T))
print(c)