import numpy as np
import matplotlib.pyplot as plt

# Create a dataset
X = np.array([
    [2, 3],
    [1, 1],
    [2, 1],
    [3, 2],
    [7, 8],
    [8, 9],
    [9, 7]
])

y = np.array([-1, -1, -1, -1, 1, 1, 1])

# ----------------------------------------------------------------
# Create a linear SVM classifier
# ----------------------------------------------------------------
class SimpleSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1
        self.w = np.zeros(n_features) # Initialize weights
        self.b = 0 # Initialize bias

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    
# ----------------------------------------------------------------
# Train the SVM classifier
# ----------------------------------------------------------------
svm = SimpleSVM()
svm.fit(X, y)

predictions = svm.predict(X)
print("Predictions:", predictions)

# Visualize the decision boundary
for idx, point in enumerate(X):
    plt.scatter(*point, color='red' if y[idx] == -1 else 'blue', marker='x')

w = svm.w
b = svm.b
x0 = np.linspace(0, 4, 10)
x1 = -(w[0] * x0 - b) / w[1]  # 解线性方程 w·x - b = 0
plt.plot(x0, x1, '--')
plt.title("SVM Decision Boundary")
plt.show()