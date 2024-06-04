import numpy as np
import matplotlib.pyplot as plt

class LinReg(object):
    def __init__(self):
        self.m = None
        self.b = None

    def predict(self, X):
        if self.m is None or self.b is None:
            raise Exception("Model not fitted yet")
        return self.m * X + self.b

    def get_params(self):
        if self.m is None or self.b is None:
            raise Exception("Model not fitted yet")
        return self.m, self.b

    def plot_regression_line(self, X, y):
        plt.scatter(X, y)
        print(f"The solution is y = {self.m}*X + {self.b}")
        plt.plot(X, self.predict(X), color='red')
        plt.show()

    def Score(self, X_predict, y_true):
        y_pred = self.predict(X_predict)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def fit(self, X, y):
        self.m = (len(X) * np.sum(X * y) - np.sum(X) * np.sum(y)) / (len(X) * np.sum(X ** 2) - np.sum(X) ** 2)
        self.b = (np.sum(y) - self.m * np.sum(X)) / len(X)