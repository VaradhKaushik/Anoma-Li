# Code for Isolation Forest as anomaly detection

import numpy as np
import pandas as pd


class IForest():

    def __init__(self, df, max_dep=25, seed=14, *args, **kwargs):
        self.df = df
        self.max_dep = max_dep
        self.seed = seed
    
    def __str__(self):
        return f"\nIsolation Forest object with max_depth: {self.max_dep} and seed: {self.seed}\n"
    
    

df = [1,2,3]

test = IForest(df)
print(test)
