import numpy as np

X = np.array(##データセット##)

# 標準化
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 相関行列
R = np.corrcoef(X.T)


# 固有値分解
eigen_values, eigen_vectors = np.linalg.eigh(R)

# 特徴変換
X_pca = X.dot(eigen_vectors)
