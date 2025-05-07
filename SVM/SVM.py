from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

######### Load the breast cancer dataset #########
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(X.head())
print(y.head())

######### Visualize the data #########
df = X.copy()
df['target'] = y
print(df.head())

sns.scatterplot(data=df, x="mean radius", y="mean texture", hue="target", palette="Set1")

plt.title("mean radius vs. mean texture by diagnosis")
# plt.show()

######### PCA Dimensionality Reduction #########
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) #
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1")
plt.title("PCA of Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()