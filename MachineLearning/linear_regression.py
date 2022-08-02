import random

import numpy as np
import pandas as pd


class LinearRegession:
    def __init__(self, X=None, Y=None, seed=2022):
        if not X or not Y:
            self.get_X_y()
        else:
            self.X = X
            self.Y = Y
        self.n_samples, self.n_features = self.X.shape
        self.init_w(self.n_features)
        self.data_normalization()
        self.seed_all(seed)
        self.loss_history = []

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_X_y(self):
        # X, y = load_boston(return_X_y=True)
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        target = target.reshape((-1, 1))
        self.X = data
        self.Y = target

    def get_loss(self, y_pred, y):
        return 1 / y.shape[0] * np.linalg.norm(y_pred - y) * 1 / 2

    def init_w(self, shape):
        self.w = np.random.randn(shape, 1) * 0.01
        self.w = np.concatenate((np.ones((1, 1)), self.w), axis=0)

    def data_normalization(self):
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.X = np.concatenate((np.ones((self.n_samples, 1)), self.X), axis=1)

    def train(self, batch_size=1, epoches=400, lr=0.01, method="SGD"):
        self.loss_history = []
        for i in range(1, epoches + 1):
            y_pred = np.dot(self.X, self.w)
            loss = self.get_loss(y_pred, self.Y)
            self.loss_history.append(loss)
            if not i % 10:
                print(f"---- epoch {i} / {epoches} ----")
                print(f"Loss: {round(loss, 3)}")

            if method == "SGD":
                for j in range(0, self.n_samples, batch_size):
                    diff_rad = y_pred[j: j + batch_size] - self.Y[j: j + batch_size]
                    dw = (self.X[j: j + batch_size].T.dot(diff_rad))/ self.n_samples
                    self.w = self.w - lr * dw

                pre = self.n_samples // batch_size * batch_size
                if pre != self.n_samples:
                    diff_rad = y_pred[pre:] - self.Y[pre:]
                    dw = (self.X[pre:].T.dot(diff_rad)) / self.n_samples
                    self.w = self.w - lr * dw

            elif method == "GD":
                diff = y_pred - self.Y
                dw = self.X.T.dot(diff) / self.n_samples
                self.w = self.w - lr * dw

    def predict(self, x):
        return (self.w[0] + np.dot(x, self.w[1:]))[0]

    def plot_loss_history(self):
        from matplotlib import pyplot as plt
        plt.plot(self.loss_history)
        plt.xlabel("Epoches")
        plt.ylabel("Loss")
        plt.show()


lr = LinearRegession()
lr.train(method="SGD", batch_size=20)
print(lr.predict(np.random.random(13)))
