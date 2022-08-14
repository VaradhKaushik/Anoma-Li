# Anoma-Li
Anomaly detection is a critical tool in identifying abnormal behavior in a data set because anomalies and outliers show that something unexpected has occurred.

Anoma-Li aims to give an end-to-end solution to perform anomaly detection, from creating the dataset to visualizing the anomalies.Choose from various types of anomaly detection algorithms to perform outlier detection, and visualize the anomalies in the form of graphs.

The anomaly detection methods that this package performs are:
- K Means Classification
- Isolation Forest
- K-NN Classification
- Mean Average Deviation

## Getting Started
### Dependencies
* Python3
    * Numpy
    * Pandas
    * Matplotlib
    * Sklearn [For test dataset]

### Running the code
* Clone the repo to your local machine
* Run main.py to see how all the algorithms perform on synthetic data with visualization

OR

* Run each algorithm individually by:
    * IForest:
```
# Importing IForest
from iforest import *

# Make anomaly detection on a DataFrame
iforest_pred(n=100, cntm=0.05, subspace=256, df=None, seed=14)

# Doesn't return anything, adds a column to DataFrame called "IF_anomaly", 
# which contains binary values for anomalies [0-Normal point, 1-Anomaly]
```

*   * K-means clustering

```
# Importing K-means
from kmeans import *

# Create instance of Kmeans class and pass parameters
obj = Kmeans(X, 5, 100)

# Calculate distances
obj.Calc_dic()

# Output the value of outliers
print(obj.get_cluster_data(data))
print(obj.get_outliers(data))

# Plots the anomalies
obj.WCSS_graph()
```

*   * K-Nearest Neighbors

```
# Importing KNN
from knn import *

# Create KNN object
knn = KNN(k, xtrain, ytrain, xtest)

# Get anomalies
anomalies = knn.detect_anomaly(df, threshold)
```

*   * Mean

