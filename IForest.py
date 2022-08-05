    # Code for Isolation Forest as anomaly detection
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IForestObject():

    def __init__(self, n, df, contamination, max_dep=25, subspace=24, seed=14, *args, **kwargs):
        self.n = n
        self.df = df
        self.subspace = subspace
        self.contamination = contamination
        self.max_dep = max_dep
        self.seed = seed
        np.random.seed(seed)
    
    # def __str__(self):
        # return f"\nIsolation Forest object with: {self.n} trees, max_depth: {self.max_dep} and seed: {self.seed}\n"
    
    
    def feature_selection(self, df):
        """
        Selects random feature from dataframe
        """
        return random.choice(df.columns)
    

    def cutoff_value(self, df, feature):
        """
        Selects random cutoff value of a feature/column
        """
        min_i = df[feature].min()       # Min value of selected feature
        max_i = df[feature].max()       # Max value of selected feature

        # ASSERT HERE

        return (max_i - min_i) * np.random.random() + min_i     # Random value b/w Min and Max of feature
    

    def partition_feature(self, df, feature, val):
        """
        Split feature based on cutoff value
        """
        data_1 = df[df[feature] <= val]
        data_2 = df[df[feature] > val]

        return data_1, data_2
    

    def classify_data(self, df):
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        label_col = df.values[:, -1]    # df specific
        uq_classes, cnt_uq_classes = np.unique(label_col, return_counts=True)
        
        ix = cnt_uq_classes.argmax()    # index of class with most occurences
        classification = uq_classes[ix]

        return classification
    
    def ITree(self, df, cnt=0):

        # Termination case
        if (cnt == self.max_dep) or (df.shape[0] <= 1):     # If max depth reached or final value in df
            return self.classify_data(df)

        else:
            cnt += 1

            split_column = self.feature_selection(df)
            split_value = self.cutoff_value(df, split_column)

            data1, data2 = self.partition_feature(df, split_column, split_value)

            # Instantiate sub-tree
            question = f"{split_column} <= {split_value}"
            sub_tree = {question: []}

            # Recursion
            ans1 = self.ITree(data1, cnt)
            ans2 = self.ITree(data2, cnt)

            if ans1 == ans2:
                sub_tree = ans1
            
            else:
                sub_tree[question].append(ans1)
                sub_tree[question].append(ans2)
            
            return sub_tree
            
    def IForest(self): 
        forest = []
        
        for i in range(self.n):

            if self.subspace <= 1:
                data = self.df.sample(frac=self.subspace)
            
            else:
                data = self.df.sample(self.subspace)
        
            # Fitting Tree
            tree = self.ITree(data)


            # Add tree to forest
            forest.append(tree)
        
        return forest


    def pathLength(self, example, itree, path=0):

        path += 1
        question = list(itree.keys())[0]

        feat_name, comp_oprt, value = question.split()


        if example[feat_name].values <= float(value):
            ans = itree[question][0]
        else:
            ans = itree[question][1]

        # terminal case
        if not isinstance(ans, dict):
            return path
        
        # recursive part
        else:
            residual_tree = ans
            return self.pathLength(example=example, itree=residual_tree, path=path)

        return path


    def evaluate_instance(self, instance, forest):
        paths = []
        for tree in forest:
            paths.append(self.pathLength(instance, tree))
        return paths


    def c_factor(self, n):
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


    def anomaly_score(self, data_point,forest,n):
        '''
        Anomaly Score
        
        Returns
        -------
        0.5 -- sample does not have any distinct anomaly
        0 -- Normal Instance
        1 -- An anomaly
        '''
        # Mean depth for an instance
        E = np.mean(self.evaluate_instance(data_point,forest))
        c = self.c_factor(n)
        
        # return E
        return 2**-(E/c)


##### Generating Test Dataset #####
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
Nobjs = 2000

x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
x[0], y[0] = 3.3, 3.3       #Add manual outlier

df=np.array([x,y]).T
df = pd.DataFrame(df,columns=['feat1','feat2'])

plt.figure(figsize=(7,7))
plt.plot(x,y,'bo')
plt.show()

##### Algorithm called from here #####
anms = IForestObject(n=10, df=df, contamination=0.05)
trees = anms.IForest()

an= []
for i in range(df.shape[0]):
    an.append(anms.anomaly_score(data_point=df.iloc[[i]], forest=trees, n=25))

ans = np.array(an)
print(np.unique(ans))

plt.hist(an)
plt.show()
