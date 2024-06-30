import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class SelectFeatures:
    def __init__(self, cfg):
        self.random_state = cfg.random_state
        
    def __call__(self, X, y):
        """
        Selects the most important features based on three criteria:
        1. Random Forest feature importance
        2. Lasso (L1 regularization) coefficients
        3. Custom technique (to be implemented)
        
        Args:
        X (np.array): Input features
        y (np.array): Target variable
        
        Returns:
        list: Indices of selected features
        """
        rf_features = self._random_forest_selection(X, y)
        lasso_features = self._lasso_selection(X, y)
        custom_features = self._custom_selection(X, y)
        print(rf_features, lasso_features)


    def _random_forest_selection(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        return importances
    
    def _lasso_selection(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = Lasso(random_state=42, alpha=5000, positive=True, fit_intercept=False, max_iter=1000)
        lasso.fit(X_scaled, y)
        if not all(lasso.coef_ == 0):
            return lasso.coef_ / np.sum(lasso.coef_)
        return lasso.coef_
    
    def _custom_selection(self, X, y):
        # Generate 3 new random features
        n_samples = X.shape[0]
        random_feature1 = np.random.normal(0, 1, n_samples)
        random_feature2 = np.random.normal(5, 2, n_samples)
        random_feature3 = np.random.normal(-3, 1.5, n_samples)
        
        random_df = pl.DataFrame({
        'random_feature1': random_feature1,
        'random_feature2': random_feature2,
        'random_feature3': random_feature3
        })
    
        X_extended = X.hstack(random_df)

        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_extended, y)
        
        importances = rf.feature_importances_
        
        # Find the maximum importance score of the dummy features
        dummy_importance = max(importances[-3:])
        
        # Get indices of features less important than the dummy features
        less_important_indices = np.where(importances[:-3] < dummy_importance)[0]
        
        # Print the names of less important features
        less_important_features = np.array(X.columns)[less_important_indices]
        print("Features less important than random noise:")
        for feature in less_important_features:
            print(f"- {feature}")
        