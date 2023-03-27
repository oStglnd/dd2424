

import numpy as np

class linearClassifier:
    def __init__(self, K, d):
        self.K = K
        self.d = d
        
        # init weights
        W = np.random.normal(
            loc=0, 
            scale=0.1, 
            size=(self.K, d)
        )
        
        # init bias
        b = np.random.normal(
            loc=0, 
            scale=0.1, 
            size=(self.K, 1)
        )

    def evaluate(self, X):
        
        return 0
    
    def computeCost(self, X, Y, lambd):
        
        return 0
    
    def computeAcc(self, X, y):
        
        return 0
    
    def computeGrads(self, X, Y, P, lambd):
        
        return 0

