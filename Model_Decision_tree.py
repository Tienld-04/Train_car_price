import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MyDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    class Node:
        __slots__ = ('feature', 'threshold', 'left', 'right', 'value')
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        preds = [self._predict_row(x, self.root) for x in X]
        return np.array(preds)

    def _build_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split) or np.unique(y).size == 1:
            return self.Node(value=np.mean(y))

        best_feat, best_thresh, best_score = None, None, np.inf
        current_mse = np.var(y) * n_samples

        for feature in range(n_feats):
            thresholds = np.unique(X[:, feature])
            for thr in thresholds:
                left_mask = X[:, feature] <= thr
                if left_mask.sum() < self.min_samples_split or (~left_mask).sum() < self.min_samples_split:
                    continue
                y_left, y_right = y[left_mask], y[~left_mask]
                score = (np.var(y_left) * y_left.size) + (np.var(y_right) * y_right.size)
                if score < best_score:
                    best_score, best_feat, best_thresh = score, feature, thr

        if best_feat is None or current_mse - best_score < 1e-7:
            return self.Node(value=np.mean(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)
