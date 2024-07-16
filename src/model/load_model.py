import torch
import os
import json
import pickle
import joblib
from cachetools import cached, LFUCache
from src.model.ensemble import Ensemble

class ModelLoader:
    def __init__(self, cfg, k=1):
        self.cfg = cfg
        self.k = k
    
    @staticmethod
    def load_model(filename):
        if filename.endswith(('.pth', '.pt')):
            checkpoint = torch.load(filename)
            model = checkpoint['model']
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            return model
        elif filename.endswith(('.pkl')):
            with open(filename, 'rb') as f:
                return pickle.load(f) 
        elif filename.endswith(('.joblib')):
            return joblib.load(filename)
        else:
            raise ValueError(f"Unsupported model file extension: {filename}")
    
    @cached(cache=LFUCache(maxsize=16))
    def load(self, figi: str) -> Ensemble:
        """Loads the top k models based on ensemble scores or recency."""
        model_path = os.path.join(self.cfg.paths.models, figi)
        ensemble_file = self._get_latest_ensemble_data(model_path)

        weights = []
        models = []
        if ensemble_file:
            with open(os.path.join(model_path, ensemble_file), 'r') as f:
                ensemble_data = json.load(f)

            sorted_models = sorted(ensemble_data['models'], key=lambda x: x['score'], reverse=True)
            top_k_models = sorted_models[:self.k]

            for model_info in top_k_models:
                model = ModelLoader.load_model(os.path.join(model_path, model_info['filename']))
                models.append(model)
                if 'weight' in model_info:
                    weights.append(model_info['weight'])
                else:
                    weights.append(1 / self.k)
        else:
            model_files = sorted(os.listdir(model_path), key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)
            for file in model_files[:self.k]:
                model = ModelLoader.load_model(os.path.join(model_path, file))
                models.append(model)
            weights = [1 / self.k] * self.k
        return Ensemble(models, weights)

    def _get_latest_ensemble_data(self, model_path):
        """Returns the filename of the most recent ensemble file ('metadata.json')."""
        ensemble_filename = "metadata.json" 
        ensemble_filepath = os.path.join(model_path, ensemble_filename)
        if os.path.exists(ensemble_filepath):
            return ensemble_filename
        return None 