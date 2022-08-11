import pandas as pd
from collections import Counter


class KNN:
    
    
    def __init__(self, k, xtrain, ytrain, xtest):
        '''
        Initialises the KNN class

        Parameters
        ----------
        k : Number of neighbours that need to be considered when deciding class.
        
        xtrain : Training values of the paramenters in a pandas DataFrame.
        
        ytrain : Training values of class in a pandas DataFrame.
        
        xtest : Testing values of parameters in a pandas DataFrame.

        Returns
        -------
        None.

        '''
        self.k = k
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        
    def euclidist(self, x,y):
        '''
        Finds the euclidean distance between 2 points

        Parameters
        ----------
        x : Iterable containing dimension values of point 1.
        
        y : Iterable containing dimension values of point 2.

        Returns
        -------
        temp : Euclidean distance

        '''
        temp = 0
        
        for i in x:
            for j in y:
                temp += (j-i)**2
                
        temp = temp**0.5
        return temp
    
    def predict1(self, test_pt):
        '''
        Predicts the class for a single datapoint

        Parameters
        ----------
        test_pt : Iterable containing dimensions of the point to be tested.

        Returns
        -------
        label : Class of tyhe point.

        '''
        distlist = []
        
        for i in self.xtrain.index:
            
            distlist.append(self.euclidist(test_pt, self.xtrain.loc[i].values))
        dists_df = pd.DataFrame(data=distlist, index=self.xtrain.index, columns=['dist'])
        dists_df = dists_df.sort_values(by=['dist'], axis=0)[:self.k]
        temp = []
        for i in dists_df.index.values:
            temp.append(self.ytrain[i])
        counter = Counter(temp)
        
        label = counter.most_common(1)[0][0]
        return label
    
    def predict(self):
        '''
        Predicts classes for an entire DataFrame of test points

        Returns
        -------
        ytest : pandas DataFrame containing the test points.

        '''
        ytest_list = []
   
        for test_index in self.xtest.index:
            
            test_pt = self.xtest.loc[test_index].values
            
            pt = self.predict1(test_pt.tolist())
            ytest_list.append(pt)
        ytest = pd.DataFrame(data = ytest_list, index = self.xtest.index, columns = ['target'])
        return ytest
    
