import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

wine = load_wine()
X = wine.data
y = wine.target
df_X = pd.DataFrame(X, columns=wine.feature_names)
df_y = pd.DataFrame(y, columns=['target'])

print(df_X.describe())
print(df_X.dtypes)
print(df_y.describe())

scaler = StandardScaler()
X_std = scaler.fit_transform(X)



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)



knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear')

knn.fit(X_pca, y)
svm.fit(X_pca, y)

y_pred_knn = knn.predict(X_pca)
y_pred_svm = svm.predict(X_pca)
print(f"Accuracy KNN: {accuracy_score(y, y_pred_knn):.2f}")
print(f"Accuracy SVM: {accuracy_score(y, y_pred_svm):.2f}")



fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax[0].contourf(xx, yy, Z, alpha=0.2)
ax[0].set_title('KNN')

ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y)
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax[1].contourf(xx, yy, Z, alpha=0.2)
ax[1].set_title('SVM')

plt.show()