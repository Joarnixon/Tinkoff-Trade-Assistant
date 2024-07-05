from sklearn.linear_model import LogisticRegression
from src.model.base_model import Model
import torch.nn as nn
import torch.nn.functional as F
from numpy import max
from torch import argmax, tensor, float32

class LinearModel(Model):
    def __init__(self, args):
        self.model = LogisticRegression(**args)
        
    def predict(self, X):
        return self.model.predict(X), self.model.predict_proba(X)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

class LinearNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_layers), nn.ReLU(), nn.Linear(hidden_layers, output_size))

    def forward(self, x):
        preds = self.net(x)
        return preds, F.softmax(preds, dim=1)
    
    def predict(self, x):
        x = tensor(x, dtype=float32).squeeze(0)
        preds, probas = self.forward(x)
        return argmax(preds, dim=1).cpu().numpy(), probas.detach().cpu().numpy()