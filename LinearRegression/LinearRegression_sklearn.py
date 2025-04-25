from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 准备数据
# X 是特征，y 是目标变量
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # 实际上是 y = 2 * x

# 2. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 预测和评估
y_pred = model.predict(X_test)
print("预测结果：", y_pred)
print("均方误差：", mean_squared_error(y_test, y_pred))

# 5. 输出模型参数
print("模型系数：", model.coef_)
print("模型截距：", model.intercept_)
