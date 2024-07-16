import numpy as np
import polars as pl
import json
import os
from typing import Optional, Union
from src.data.data_manager import DataManager, NullDataManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class SelectFeatures:
    def __init__(self, cfg, data_manager: Optional[DataManager] = None):
        self.random_state = cfg.random_state
        self.data_manager = data_manager if data_manager is not None else NullDataManager()
        if os.path.isfile(cfg.paths.selected_features):
            with open(cfg.paths.selected_features, 'r') as f:
                self.preselected_features = json.load(f)
        else:
            self.preselected_features = {}

    def fit_transform(self, X, y, figi: Optional[str] = None):
        """
        Selects the most important features based on two criteria:
        1. Random Forest feature importance
        2. Random noise importance comparison
        """
        time_column = X.select(['time']).to_series()
        X = X.drop(['time'])
        custom_features = self.compare_with_random_noise(X, y)
        k = len(custom_features)
        rf_features = self._random_forest_selection(X, y, k)
        # lasso_features = self._lasso_selection(X, y)
        # print(lasso_features)

        features_to_remove = set(rf_features).intersection(set(custom_features))
        columns_to_keep = [col for i, col in enumerate(X.columns) if i not in features_to_remove]
        X_selected = X.select(columns_to_keep)

        self.data_manager.write_selected_features(columns_to_keep, figi)
        return X_selected.insert_column(0, time_column)
    
    def fit_transform_online(self, X: pl.DataFrame, figi: Optional[str] = None):
        if figi not in self.preselected_features:
            return X.drop('time')
        return X.select(self.preselected_features[figi])

    def _random_forest_selection(self, X, y, k):
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        return np.argsort(importances)[:k]
    
    def _lasso_selection(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = Lasso(random_state=self.random_state, alpha=5000, positive=True, fit_intercept=False, max_iter=1000)
        lasso.fit(X_scaled, y)
        if not all(lasso.coef_ == 0):
            return lasso.coef_ / np.sum(lasso.coef_)
        return lasso.coef_
    
    def compare_with_random_noise(self, X, y):
        n_samples = X.shape[0]
        np.random.seed(self.random_state)
        random_feature1 = np.random.normal(0, 1, n_samples)
        random_feature2 = np.random.normal(5, 2, n_samples)
        random_feature3 = np.random.normal(10, 10, n_samples)
        
        random_df = pl.DataFrame({
            'random_feature1': random_feature1,
            'random_feature2': random_feature2,
            'random_feature3': random_feature3
        })
    
        X_extended = X.hstack(random_df)

        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
        rf.fit(X_extended, y)
        
        importances = rf.feature_importances_
        dummy_importance = max(importances[-3:])
        
        less_important_indices = np.where(importances[:-3] < dummy_importance)[0]
        
        # less_important_features = np.array(X.columns)[less_important_indices]
        
        # print("Features less important than random noise:")
        # for feature in less_important_features:
        #     print(f"- {feature}")
        
        return less_important_indices