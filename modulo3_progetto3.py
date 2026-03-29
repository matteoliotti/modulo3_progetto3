import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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