import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd


class Kmeans:

    '''
    In this class, K-means algorithm is used to find outliers in the input dataset
    K-means is an Unsupervised Machine Learning algorithm. It is used to classifying data into clusters.
    The input dataset is unlabeled and the model figures out the clusters through multiple iterations.
    Initially, random values are selected as centroids. Through each iteration, the value of Centroids are updated based
    on the distance of each point to the centroids. Each point is assigned to its nearest Centroid,thus forming clusters.
    In this class, outliers are detected from the dataset based on the maximum distance from final Centroids.
    Using functions defined within this class,Outliers from the dataset and Cluster label for each point can be obtained.
    '''

    def __init__(self, X, k, iterations):

        """
        init function for Kmeans outlier detection class

        :param X: Input data columns; type - dataframe

        :param k: number of clusters

        :param iterations: number of iterations
        """

        def init_centroid(m, n, K, X):

            """
            Function initializes the value  for Centroid. random.randint is used to select a random points from the input data points. These values are used as initial Centroids

            :param : m is the number of rows in input. n is the number of columns.
            For a 2D array, n=2. K is the nuber of clusters. X is the input data. Return - returns initial values for
            Centroid. Return type is Array.

            :return: returns initial values for Centroid. Return type is Array.
            """
            Centroids = np.array([]).reshape(n, 0)
            for i in range(K):
                rand = rd.randint(0, m - 1)
                Centroids = np.c_[Centroids, X[rand]]
            return Centroids.T

        self.iterations = iterations
        self.outliers1 = None
        self.K = k
        self.X_DF=X
        self.X = X.iloc[:].values
        self.temp_dist_arr = None
        self.m = X.shape[0]  # number of training examples
        self.n = X.shape[1]  # number of features. Here n=2
        self.Centroids = init_centroid(self.m, self.n, self.K, self.X)
        # object.Centroids gives the Centroids for the object
        self.dic_clus = {}

    def get_avg(self):

        """
        Function get_avg calculates the average of points assigned to each cluster
        This result is  the updated values for Centroid array in each iteration.

        :return: Centroid array of size k X n
        """
        Centroids_t = self.Centroids.T
        for key in range(self.K):
            Centroids_t[:, key] = np.mean(self.dic_clus[key], axis=0)
        return Centroids_t.T

    def max_distance(self, arr):
        """
        Calculates outliers based on distance of each point from its closest centroid.

        :param arr: arr is mXk array containing distance of each point to each of the Centroids.

        :return: None
        """
        min_arr_new = []
        ind = np.argmin(arr, axis=1)
        for i in range(self.m):
            min_arr_new.append(arr[i][ind[i]])
        X_maxdis_ar = np.vstack((self.X.T, min_arr_new)).T

        panda_df = pd.DataFrame(data=X_maxdis_ar, columns=[list(self.X_DF.columns)[0], list(self.X_DF.columns)[1], "Distance"])
        self.outliers1 = panda_df.sort_values('Distance', ascending=False).head(int(0.04 * self.m))
        return None

    def update_centroids(self):
        """
        Updates the values for Centroids.

        :return: None
        """
        self.Centroids = self.get_avg()
        return None

    def Calc_dic(self):
        """
        Calc_dic function calculates the distance of each point to each Centroid. Depending on minimum distance,
        each point is then assigned to a centroid using a dictionary. update_centroids function is then called to update
        the Centroids based on the new values in dictionary. Calc_dic is called after creating instance of class Kmeans

        :return: None
        """
        for itr in range(self.iterations):
            dis_arr = np.array([]).reshape(self.m, 0)
            for t in range(0, self.K):
                temp_dis_arr = (np.sum(((self.X - self.Centroids[t]) ** 2), axis=1) ** (1 / 2))
                dis_arr = np.c_[dis_arr, temp_dis_arr]
                self.temp_dist_arr = dis_arr

            index = np.argmin(dis_arr, axis=1)
            colm = []
            min_arr = []
            for i in range(self.m):
                min_arr.append(dis_arr[i][index[i]])
                colm.append(index)
            min_val_idx = np.vstack((min_arr, colm)).T

            for i in range(self.K):
                self.dic_clus[i] = []

            for i in range(len(dis_arr)):
                self.dic_clus[int(min_val_idx[i][1])].append(self.X[i])

            self.update_centroids()
        self.max_distance(self.temp_dist_arr)

    def cluster_2xm_array(self, key):
        """
        Returns 2Xm array of X and Y points taken from each value in the dictionary.

        :param key: key = self.key is the number of clusters.

        :return: 2Xm array
        """
        l1 = []
        l2 = []
        if len(self.dic_clus[key]) != 0:
            for n in self.dic_clus[key]:
                l1.append(n[0])
                l2.append(n[1])
            arr = np.array([l1, l2])
            return arr
        if len(self.dic_clus[key]) == 0:
            ar = np.array([0][0])
            return ar

    def get_cluster_data(self, data):
        """
        Appends cluster value to input dataframe.

        :param data: data is the input dataframe containing all rows and columns from Excel sheet.

        :return: returns dataframe with cluster values appended
        """
        index = np.argmin(self.temp_dist_arr, axis=1)
        index = index + 1
        df = pd.DataFrame({'cluster_label': index})
        df_join = pd.concat([data, df], axis=1, join='inner')
        return df_join

    def get_outliers(self, data):
        """
        Calculates outlier points in the input. It Returns dataset along with the classification value of each data point

        :param data: data is the input dataframe containing all rows and columns from Excel sheet.

        :return: Returns dataframe containing only the outliers
        """
        res1 = pd.merge(data, self.outliers1, on=[list(self.X_DF.columns)[0],list(self.X_DF.columns)[1] ])
        return res1

    def cluster_list(self,p):
        """
        Function to convert each value in dictionary containing clusters to 2 lists (X,Y) for scatter plot.

        :return: two lists for scatter plot
        """
        l1 = []
        l2 = []
        if len(self.dic_clus[p]) > 0:
            for r in range(len(self.dic_clus[p])):
                l1.append(self.dic_clus[p][r][0])
                l2.append(self.dic_clus[p][r][1])
        return l1,l2

    def WCSS_graph(self):
        """
        Iterates values of number of clusters from 1 to 10. Used to identify the appropriate value of k needed.

        :return: Returns a graph of Within-Cluster Sums of Squares vs number of clusters
        """
        WCSS_arr = np.array([])
        for k in range(2, 11):
            self.__init__(self.X_DF, k, 100)
            self.Calc_dic()
            Centroidz = self.Centroids
            wcss = 0

            for k in range(self.K):
                clus_arr = self.cluster_2xm_array(k)
                wcss = wcss + np.sum((clus_arr.T - Centroidz[k, :]) ** 2)
            WCSS_arr = np.append(WCSS_arr, wcss)
        print("WCSS_arr = ", WCSS_arr)

        K_array = np.arange(2, 11, 1)
        plt.plot(K_array, WCSS_arr)
        plt.xlabel('Number of Clusters')
        plt.ylabel('within-cluster sums of squares (WCSS)')
        plt.title('Elbow method to determine optimum number of clusters')
        plt.show()

