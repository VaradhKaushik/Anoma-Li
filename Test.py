import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

import Kmeans_outlier_detection as km

#Starting unit test for Kmeans class
class Unittest_kmeans:
    """
    This class is te unit test for Kmeans Classification and Outlier Detection package.
    It validates the type and shape of the results (array or dataframe).
    It also uses a synthesized data set to validate that the predicted classification is equal to the actual class of each data point
    """
    X, y = make_blobs(n_samples=200, centers=5, n_features=2)
    data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    data.columns = ['X_values', 'Y_values', 'label']
    X = data[["X_values", "Y_values"]]

    obj = km.Kmeans(X, 5, 100)
    obj.Calc_dic()

    obj.get_cluster_data(data)

    assert obj.Centroids.shape == (5, 2), f"shape (5,2) is expected, got: {obj.Centroids.shape}"
    print("Shape of Centroid array is as expected")
    assert (obj.get_cluster_data(data)).shape == (200, 4), f"shape (200, 4) is expected, got: {(obj.get_cluster_data(data)).shape}"
    print("Shape of cluster dataframe is as expected")
    assert (obj.get_outliers(data)).shape == (8, 4), f"shape (8, 4) is expected, got: {(obj.get_outliers(data)).shape}"
    print("Shape of outliers dataframe is as expected")

    dt = {
        "X_values": [5.413135179, 5.624100153, 6.837685531, -4.102930791, 6.769483915, -4.087144278, -3.978928539,
                     -5.811959514, 4.725200243, 7.018239141, -6.280291711, 5.574055729, -5.145636717, 5.084600549,
                     -5.641834146, -5.066427339, -4.96374198, 4.727492566, 6.863669071, 4.779772274, -4.946931331,
                     4.700060785, -2.988560311, -3.551831812, -4.019779264, -6.155789571, 6.605564165, -2.611218367,
                     6.2676015, 5.560042787, 6.86176286, -5.215997626, -3.884820203, 4.721171513, -5.462341487,
                     -4.126857263, 5.160441757, -6.006500466, -4.969213404, -4.355738409, -4.14475138, -4.42985766,
                     -3.950854424, 7.073474556, -5.281645443, 7.632245897, -3.905701822, 5.182938976, -4.113797686,
                     -4.065160339, 6.396841414],
        "Y_values": [-7.196390063, -6.168104013, -4.693934183, -7.333215341, -5.420551724, -6.900783848, -7.308746131,
                     -6.962860496, -6.113208564, -5.694848859, -6.264615182, -6.186441238, -7.526537881, -7.363255714,
                     -5.863770417, -8.505738965, -6.744344904, -6.161557739, -5.279224427, -5.225046954, -5.85764687,
                     -3.599815178, -5.953636352, -6.422813702, -6.685124968, -7.454110175, -4.979516312, -7.391182901,
                     -4.502911705, -3.727736758, -4.900436652, -7.559859092, -9.648850992, -5.668539418, -7.704721065,
                     -6.49423689, -6.511215654, -8.238000608, -6.876909761, -8.579393695, -6.574297148, -6.591972827,
                     -7.049482539, -4.417786281, -6.289807327, -5.467683036, -8.386040114, -8.561267915, -9.051416929,
                     -6.160346619, -4.819828868],
        "label": [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
                  0,
                  0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    }
    data2 = pd.DataFrame(dt)
    X2 = data2[["X_values", "Y_values"]]

    obj2 = km.Kmeans(X2, 2, 100)
    obj2.Calc_dic()

    clus = obj2.get_cluster_data(data2)

    assert np.array_equal(np.array(clus["label"]) + 1, np.array(clus["cluster_label"])) or np.array_equal(
        np.array(clus["label"]), np.absolute(np.array(clus["cluster_label"]) - 2)), "Clustering NOT as predicted."
    print("Classification of data2 points is correct")
    print("All Tests have been completed for KMEANS clustering and outlier detection algorithm")
    print("Plotting sample data...")
    
    # End of unit test for Kmeans class

