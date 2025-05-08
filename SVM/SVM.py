##################################################################
############## Data Preprocessing and Visualization ##############
##################################################################
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

# print(X.head())
# print(y.head())

######### Visualize the data #########
df = X.copy()
df['target'] = y
# print(df.head())

# sns.scatterplot(data=df, x="mean radius", y="mean texture", hue="target", palette="Set1")
# plt.title("mean radius vs. mean texture by diagnosis")
# plt.show()

######### PCA Dimensionality Reduction #########
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) #
pca = PCA(n_components=2) # Create a PCA object to reduce to 2 dimensions   
X_pca = pca.fit_transform(X_scaled) # 

# plt.figure(figsize=(8,6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1")
# plt.title("PCA of Breast Cancer Dataset")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# splt.show()

##################################################################
############# SVM Classifier Based on Data after PCA #############
##################################################################
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Create a SVM classifier
svm = SVC(kernel='linear', C=1.0) # Create a SVM classifier with linear kernel
svm.fit(X_train, y_train) # Fit the model to the training data
svm_predictions = svm.predict(X_test) # Make predictions on the test data

# Print the classification report
print(classification_report(y_test, svm_predictions))
# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, svm_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix")   
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

