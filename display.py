import pandas as pd
import numpy as np

class Display():
    
    def __init__(self, ypred, ytest):
        '''
        Initialises the Metrics class

        Parameters
        ----------
        ypred : DataFrame containing the predicted class values.
        
        ytest : DataFrame containing the test class values.

        Returns
        -------
        None.

        '''
        
        self.ypred = ypred
        self.ytest = ytest
        
        
        ypred_cols = list(ypred.columns)
        ytest_cols = list(ytest.columns)
        ypred.rename(columns = {ypred_cols[0]:'target'}, inplace = True)
        ytest.rename(columns = {ytest_cols[0]:'target'}, inplace = True)
        
        if len(pd.unique(ytest['target'])) >= len(pd.unique(ypred['target'])):
            self.n = len(pd.unique(ytest['target']))
        else: 
            self.n = len(pd.unique(ypred['target']))
        
        self.nmat = np.zeros((self.n, self.n),dtype = int)
    
        
                             
        for i in ypred.index:
           self.nmat[self.ytest['target'][i]][self.ypred['target'][i]] += 1 
           
           
    
    def find_accuracy(self):
        '''
        Finds the accuracy of the predicted values with respect to the class values.

        Returns
        -------
        Accuracy value

        '''
        correct_pred = 0
        for i in range(self.n):
            correct_pred += self.nmat[i][i]
        return correct_pred/np.sum(self.nmat)
    
    def confusion_matrix(self):
        '''
        Finds the confusion matrix for single or multiclass classification

        Returns
        -------
        The confusion matrix

        '''
        print(self.nmat)
        
    def precision(self, class_num):
        '''
        Finds the precision of the predicted values with respect to the test values.

        Parameters
        ----------
        class_num : value of the class for which we need to find the precision.

        Returns
        -------
        precision

        '''
        return self.nmat[class_num][class_num]/np.sum(self.nmat,axis = 0)[class_num]
        
    def recall(self, class_num):
        '''
        Finds the recall of the predicted values with respect to the test values.

        Parameters
        ----------
        class_num : value of the class for which we need to find the recall.

        Returns
        -------
        recall

        '''
        return self.nmat[class_num][class_num]/np.sum(self.nmat,axis = 1)[class_num]
    
    def f1_score(self,class_num):
        '''
        Finds the f1 score for a given class of the predicted values with respect to the test values.

        Parameters
        ----------
        class_num : value of the class for which we need to find trhe f1 score.

        Returns
        -------
        f1 score

        '''
        return (2 * self.precision(class_num) * self.recall(class_num))/(self.recall(class_num) + self.precision(class_num))
