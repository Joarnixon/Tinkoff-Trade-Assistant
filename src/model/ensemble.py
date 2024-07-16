from torch.nn import Module
import torch
from numpy import ndarray

class Ensemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X: ndarray):
        predictions = []
        probas = []
        for model, weight in zip(self.models, self.weights):
            pred, proba = model.predict(X)
            predictions.append(weight * pred)
            probas.append(weight * proba)
        return sum(predictions), sum(probas)