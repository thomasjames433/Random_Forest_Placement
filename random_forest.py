import numpy as np
from descision_tree import DescisionTree
from collections import Counter

class RandomForest():
    def __init__(self,n_trees=30,max_depth=7,min_samples_split=2,n_features=None):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_features
        self.trees=[]

    def fit(self,X,y):

        self.trees=[]
        for _ in range (self.n_trees):
            
            tree=DescisionTree(min_samples_split=self.min_samples_split,max_depth=self.max_depth, n_features=self.n_features)
            X_sample,y_sample=self._bootstrap_samples(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self,X,y):
        n_samples=X.shape[0]
        idxs=np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]
    
    def predict(self,X):
        predictions=np.array([tree.predict(X) for tree in self.trees])
        individual_preds=np.swapaxes(predictions,0,1)
        final_pred=[]
        for pred in individual_preds:
            counter=Counter(pred)
            final_pred.append(counter.most_common(1)[0][0])

        return np.array(final_pred)
