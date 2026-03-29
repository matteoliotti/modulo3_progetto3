import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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