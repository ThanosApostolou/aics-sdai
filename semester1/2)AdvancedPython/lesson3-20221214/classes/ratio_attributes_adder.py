from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RatioAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_pairs):
        self.feature_pairs = feature_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for (nom_idx, denom_idx) in self.feature_pairs:
            X = np.c_[X, X[:, nom_idx] / X[:, denom_idx]]
        return X
