import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from linear_regression import seed_all


def sigmod(y):
    return 1 / (1 + np.exp(-y))


def get_cross_entropy(y_pred, y):
    return -1 / y.shape[0] * (np.sum(y * np.log(y_pred + 1e-7) + (1 - y) * np.log(1 - y_pred + 1e-7)))


class LogisticRegression:
    def __init__(self, X=None, Y=None, seed=2022):
        if not X or not Y:
            self.get_X_y()
        else:
            self.X = X
            self.Y = Y

        self.w = None
        self.seed = seed
        self.n_samples, self.n_features = self.X.shape
        self.init_w(self.n_features)
        self.data_normalization()
        seed_all(seed)
        self.loss_history = []

    def get_X_y(self):
        self.X, self.Y = load_breast_cancer(return_X_y=True)
        self.Y = self.Y.reshape(-1, 1)

    def init_w(self, shape, plus_w0=True):
        self.w = np.random.randn(shape, 1) * 0.01
        if plus_w0:
            self.w = np.concatenate((np.ones((1, 1)), self.w), axis=0)

    def data_normalization(self, plus_w0=True):
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        if plus_w0:
            self.X = np.concatenate((np.ones((self.n_samples, 1)), self.X), axis=1)

    def train(self, batch_size=1, epoches=1000, lr=0.01, method="SGD"):
        # split train valid set
        train, valid = train_test_split(
            np.concatenate((self.X, self.Y), axis=1),
            train_size=0.7,
            random_state=self.seed,
            shuffle=True
        )

        train_X, train_Y = train[:, :-1], train[:, -1].reshape((-1, 1))
        valid_X, valid_Y = valid[:, :-1], valid[:, -1].reshape((-1, 1))

        n_samples = train_X.shape[0]
        self.loss_history = []
        for i in range(epoches):
            y_pred_train = sigmod(np.dot(train_X, self.w))
            y_pred_valid = sigmod(np.dot(valid_X, self.w))
            train_loss = get_cross_entropy(y_pred_train, train_Y)
            valid_loss = get_cross_entropy(y_pred_valid, valid_Y)
            self.loss_history.append([train_loss, valid_loss])
            if method == "SGD":
                for j in range(0, n_samples, batch_size):
                    diff_rad = y_pred_train[j: j + batch_size] - train_Y[j: j + batch_size]
                    dw = (train_X[j: j + batch_size].T.dot(diff_rad)) / n_samples
                    self.w = self.w - lr * dw

                pre = self.n_samples // batch_size * batch_size
                if pre != self.n_samples:
                    diff_rad = y_pred_train[pre:] - train_Y[pre:]
                    dw = (train_X[pre:].T.dot(diff_rad)) / n_samples
                    self.w = self.w - lr * dw

            elif method == "GD":
                diff = y_pred_train - train_Y
                dw = train_X.T.dot(diff) / n_samples
                self.w = self.w - lr * dw

            if not i % 10:
                print(f"---- epoch {i} / {epoches} ----")
                print(f"Train Loss: {round(train_loss, 3)}  Loss valid: {round(valid_loss, 3)}")

    def predict(self, x):
        return (self.w[0] + np.dot(x, self.w[1:]))[0]

    def plot_loss_history(self):
        from matplotlib import pyplot as plt
        plt.plot([i[0] for i in self.loss_history], label="train loss", color="b")
        plt.plot([i[1] for i in self.loss_history], label="valid loss", color="r")
        plt.xlabel("Epoches")
        plt.ylabel("Loss")
        plt.show()


if __name__ == '__main__':
    lg = LogisticRegression()
    lg.train(epoches=100, method="SGD", batch_size=20)
    lg.plot_loss_history()
