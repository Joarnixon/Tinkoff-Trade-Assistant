from sklearn.linear_model import LogisticRegression
from src.model.base_model import Model
from numpy import max

class LinearModel(Model):
    def __init__(self, args):
        self.model = LogisticRegression(**args)
        
    def predict(self, X):
        return self.model.predict(X), max(self.model.predict_proba(X), axis=1)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self