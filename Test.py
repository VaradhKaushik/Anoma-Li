from sklearn.datasets import make_blobs
from pandas import DataFrame

import Kmeans_outlier_detection as km

X, y = make_blobs(n_samples=200, centers=5, n_features=2)
data = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
data.columns=['X_values','Y_values','label']
X=data[["X_values","Y_values"]]

obj = km.Kmeans(X, 5, 100)
obj.Calc_dic()

assert obj.Centroids.shape == (5, 2), f"shape (5,2) is expected, got: {obj.Centroids.shape}"

assert (obj.get_cluster_data(data)).shape == (200, 4), f"shape (200, 4) is expected, got: {(obj.get_cluster_data(data)).shape}"

assert (obj.get_outliers(data)).shape == (8, 4), f"shape (8, 4) is expected, got: {(obj.get_outliers(data)).shape}"

print("All Tests have been completed for KMEANS clustering and outlier detection algorithm")