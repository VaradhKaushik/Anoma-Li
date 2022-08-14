# Code for Isolation Forest for anomaly detection
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IForestObject():

    def __init__(self, n, df, contamination, subspace=256, max_dep=25, seed=14, *args, **kwargs):
        """
        Description: __init__ for IForestObject
        
        Parameters:
            n - Number of trees in ensemble of IForest
            df - DataFrame on which anomaly detection needs to be performed
            subspace - For creation of IForest, a random slice of this size is taken from df to train each ITree
            contamination - % of anomalies in the data
            max_dep - Maximum depth to which each tree will grow
            seed - Seed for reproducibility of model
        
        Returns: None
        """
        self.n = n
        self.df = df
        self.contamination = contamination
        self.subspace = subspace
        self.max_dep = max_dep
        self.seed = seed
        np.random.seed(seed)        # Setting Numpy seed for reproducibility
    
    def __str__(self):
        """
        Setting up __str__ for the Isolation Forest Object
        """
        return f"\nIsolation Forest object with: {self.n} trees, max_depth: {self.max_dep} and seed: {self.seed}\n"
    
    
    def feature_selection(self, df):
        """
        Description: Selects random feature from dataframe

        Parameters:
            df - DataFrame from which random feature needs to be selected
        
        Returns: Random feature/column from DataFrame
        """
        return random.choice(df.columns)
    

    def cutoff_value(self, df, feature):
        """
        Description: Selects random cutoff value of a feature/column (b/w Min and Max value of feature)

        Parameters:
            df - DataFrame from which cutoff value needs to be produced
            feature - Column/feature in DataFrame from which cutoff value needs to be produced
        
        Returns: A random cutoff value b/w Min and Max value of feature
        """
        min_i = df[feature].min()       # Min value of feature
        max_i = df[feature].max()       # Max value of feature

        return min_i + (max_i - min_i) * np.random.random()     # Random value b/w Min and Max of feature
    

    def partition_feature(self, df, feature, val):
        """
        Description: Partition feature based on cutoff value

        Parameters:
            df - DataFrame which needs to be partitioned
            feature - The feature w.r.t which partion will be performed
            val - cutoff value on which the DataFrame will be split
        
        Returns: Two Partitions of the DataFrame based on the cutoff value
        """
        data_1 = df[df[feature] <= val]         # Selecting values less than or equal to the cutoff value
        data_2 = df[df[feature] > val]          # Selecting values more than the cutoff value

        return data_1, data_2                   # Returning the 2 partitions
    
    
    def itree(self, df, cnt=0):
        """
        Description: Makes an Isolation Tree using recursion.

        Parameters:
            df - Dataframe from which Isolation tree will be made.
            cnt - Current depth of the isolation tree.
        
        Returns: A nested dictionary containing the ITree
        """

        # Termination/ Base case
        if (cnt == self.max_dep) or (df.shape[0] <= 1):     # If max depth reached or final value in df
            return df.values[:, -1]

        else:
            cnt += 1                        # Increase the depth of the current ITree

            # Selecting random feature, cutoff value and partitioning
            split_column = self.feature_selection(df)
            split_value = self.cutoff_value(df, split_column)

            data1, data2 = self.partition_feature(df, split_column, split_value)

            # Storing the tree created                  # Way of storing referred from github repo cited in report
            store = f"{split_column} <= {split_value}"
            sub_tree = {store: []}

            # Recursion on the two created partitions
            ans1 = self.itree(data1, cnt)
            ans2 = self.itree(data2, cnt)

            # Storing the sub-trees (Created on partitioned data above)
            sub_tree[store].append(ans1)
            sub_tree[store].append(ans2)
            
            return sub_tree
            
    def iforest(self):
        """
        Description: Creates an ensemble of itrees to make an Isolation Forest

        Parameters:
            self - Object for which Isolation Forest has to be made. Object's parameters are used in this function.

        Returns: A list with Isolation Trees ~ Isolation Forest
        """
        forest = []
        
        for i in range(self.n):                 
            data = self.df.sample(self.subspace)    # Pick random slice of data to train trees on. 256 - ideal for huge datasets acc to Research Paper
                                                    
            # Fitting Tree
            tree = self.itree(data)

            # Add tree to forest
            forest.append(tree)
        
        return forest


    def pathLength(self, instance, itree, path=0):
        """
        Description: Recursive function that returns the path length/ depth of a particular itree

        Parameters:
            instance - A single entry/point from the dataframe
            itree - The ITree whose path length is to be found out
            path - The current path length, initialized to zero as recursively called
        
        Returns: The Path Length of a particular ITree
        """
        path += 1

        # Unpacking stored ITree
        a = list(itree.keys())[0]
        feat_name, _, value = a.split()     # "<=" won't be required so storing in "_"

        # Determining if the instance will further traverse left branch or right branch of ITree
        if instance[feat_name].values <= float(value):
            ans = itree[a][0]
        else:
            ans = itree[a][1]

        # Recursion
        if not isinstance(ans, dict):       # Termination/ Base case
            return path
        else:                               
            residual_tree = ans
            return self.pathLength(instance=instance, itree=residual_tree, path=path)


    def evaluate_instance(self, instance, forest):
        """
        Description: Returns a list of all the path lengths from all the itrees in the iforest for a particular df entry

        Parameters:
            instance - A single entry/point in the dataframe
            forest - The IForest model which has been trained
        
        Returns: A list of all the path lengths (from every tree in forest) for a point in df
        """
        paths = []
        for tree in forest:
            paths.append(self.pathLength(instance, tree))
        return paths


    def c_factor(self, n):
        """
        Description: While the maximum possible height of iTree grows in the order of n (Size of Training Data), 
        the average height grows in the order of log n. c gives the average path length of unsuccessful search in Binary Search Tree. 
        [From the IForest Research Paper Sited in report]

        Parameters:
            n - The number of external nodes [=subspace]
        
        Returns: The average path length of unsuccessful search in BST
        """
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


    def anomaly_score(self, data_point,forest, n):
        """
        Definition: Calculates an anomaly score which ranges from 0 to 1. Where:
            1    -> Point is almost certainly an anomaly
            <0.5 -> Point can safely be considered normal/ not an anomaly
            0.5  -> If all the points score ~0.5 then there are no anomalies in the data
        
        Parameters:
            data_point - The datapoint whose anomaly score is to be calculated
            forest - The trained IForest model
            n - The number of external nodes [=subspace]
        
        Returns: Anomaly score of a single data_point
        """
        # Mean depth for an instance
        E = np.mean(self.evaluate_instance(data_point,forest))
        c = self.c_factor(n)
        
        return 2**-(E/c)


##### Algorithm called from here #####
def iforest_pred(n=100, cntm=0.05, subspace=256, df=None, seed=14):
    """
    Description: This function trains an Isolation Forest model and appends the anomaly score of each entry of the dataframe to
        a column ['IF_anomaly']

    Parameters:
        n - Number of ITrees in the Isolation Forest ensemble
        cntm - Contamination of anomalies can be manually be specified with this parameter [0.05 ~ 5%]
        subspace - The size of random data taken from a DataFrame to train each tree [256 is an ideal value acc. to the research paper]
        df - The DataFrame from which the anomalies need to be found
    
    Returns: None
    """
    anms = IForestObject(n=n, df=df, contamination=cntm, subspace=subspace, seed=seed)
    trees = anms.iforest()

    an= []
    for i in range(df.shape[0]):
        an.append(anms.anomaly_score(data_point=df.iloc[[i]], forest=trees, n=subspace))

    df["IF_anomaly"] = an

    # Calculating a cutoff value from anomaly score depending on the contamination% provided
    score_list = df["IF_anomaly"].tolist()
    score_list.sort()

    ind = round(len(score_list) * cntm)
    cutoff = score_list[-ind]

    # Assigning anomaly decision based on cutoff value
    df["IF_anomaly"] = [1 if val>cutoff else 0 for val in df["IF_anomaly"]]

    return None

