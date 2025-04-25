import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # 实际上是 y = 2 * x

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance

theta_best = np.linalg.inv(X_b.T @ X_b) @X_b.T @ y  # Normal Equation

k = theta_best[1]  # 斜率
b = theta_best[0]  # 截距

sign = '+' if b >= 0 else '-'  # 符号
print(f"y = {k:.2f} x {sign} {abs(b):.2f}")  # 输出模型参数