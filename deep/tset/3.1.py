import numpy as np
import matplotlib.pyplot as plt

x = np.array([9.95, 10.14, 9.22, 8.87, 12.06, 16.30, 17.01, 18.93, 14.01, 13.01, 15.41, 14.21])
y = np.array([1.018, 1.143, 1.036, 0.915, 1.373, 1.640, 1.886, 1.913, 1.521, 1.237, 1.601, 1.496])
A = np.zeros((12, 2))

for i in range(len(x)):
    A[i] = [x[i], 1]

plt.plot(x, y, '*')
plt.show()
w = [0, 0]
A_T = A.T
w = np.linalg.inv(A_T @ A) @ A_T @ y
p = w.T @ [13.5, 1]
print(w)
print(p)
