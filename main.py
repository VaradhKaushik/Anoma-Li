from iforest import *
import pandas as pd
from collections import Counter
import numpy as np
from sklearn import datasets
import display
import matplotlib.pyplot as plt
np.random.seed(14)
def knn_display_example():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    X = df.drop('target', axis=1)
    y = df.target

    from sklearn.model_selection import train_test_split

        # Split the data - 75% train, 25% test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                           random_state=1)
    y_test = pd.DataFrame(data=y_test)
    knn = KNN(25, X_train, y_train, X_test)
    ano = knn.detect_anomaly(df,0.1)
    y_hat_test = knn.predict()
    print(ano)   
    dp = display.Display(y_hat_test, y_test)
    print('Accuracy-',dp.find_accuracy())
    print('Confusion Matrix-')
    dp.confusion_matrix()
    print('Precision-', dp.precision(1))
    print('Recall-', dp.recall(1))
    print('F1 score', dp.f1_score(1))
    plt.scatter(df["sepal length (cm)"], df["sepal width (cm)"], color = "b", s = 65)

    plt.scatter(ano["sepal length (cm)"], ano["sepal width (cm)"], color = "r")
    
##### Generating Test Dataset #####         - Cited in references
def generate_data(n=2000):
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance

    x, y = np.random.multivariate_normal(mean, cov, n).T
    x[0], y[0] = 3.3, 3.3       #Add manual outlier
    x[100], y[100] = 4.5, 3

    df=np.array([x,y]).T
    df = pd.DataFrame(df,columns=['feat1','feat2'])

    plt.figure("Generated Dataset")
    plt.title(f"Random Generated Dataset with {n} points")
    plt.xlabel("Feature 1 of df")
    plt.ylabel("Feature 2 of df")
    plt.scatter(x,y,c='turquoise', marker='o', alpha=0.7)

    return df

df = generate_data()

##### Calling the algorithms #####
iforest_pred(df=df, subspace=64, cntm=0.01, seed=14)

##### Visualization of the anomalies #####
try:
    color = ['r' if val==1 else 'g' for val in df["IF_anomaly"]]
    
    fig1 = plt.figure("Isolation Forest")
    plt.title("Isolation Forest Anomalies")
    
    plt.scatter(df['feat1'], df['feat2'], c=color, alpha=0.7)
    plt.xlabel("Feature 1 of df")
    plt.ylabel("Feature 2 of df")

except KeyError:
    print("Isolation Forest anomaly data not found")

# def visualise():

plt.show()
knn_display_example()
