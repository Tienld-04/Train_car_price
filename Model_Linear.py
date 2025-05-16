import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MyLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        X_b = np.concatenate((ones, X), axis=1)
        # theta, *_ = np.linalg.lstsq(X_b, y, rcond=None)
        result = np.linalg.lstsq(X_b, y, rcond=None)
        theta = result[0]
        self.intercept_ = theta[0]
        self.coef_      = theta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X.dot(self.coef_) + self.intercept_
