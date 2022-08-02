import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

class LogisticRegression:
    def __init__(self, X=None, Y=None, seed=2022):
        if not X or not Y:
            self.get_X_y()
        else:
            self.X = X
            self.Y = Y
        self.seed = seed
        self.n_samples, self.n_features = self.X.shape
        # self.init_w(self.n_features)
        # self.data_normalization()
        self.seed_all(seed)
        self.loss_history = []

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_X_y(self):
        self.X, self.Y = load_breast_cancer(return_X_y=True)

    def init_w(self, shape, plus_w0=True):
        self.w = np.random.randn(shape, 1) * 0.01
        if plus_w0:
            self.w = np.concatenate((np.ones((1, 1)), self.w), axis=0)

    def data_normalization(self, plus_w0=True):
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        if plus_w0:
            self.X = np.concatenate((np.ones((self.n_samples, 1)), self.X), axis=1)


lg = LogisticRegression()
print(lg.X)


