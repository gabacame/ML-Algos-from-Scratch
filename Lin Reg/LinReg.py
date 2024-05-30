import numpy as np
import matplotlib.pyplot as plt

class LinReg (object):
    def __init__(self):
        self.m = None
        self.b = None

    def predict(self, X):
        if self.m is None or self.b is None:
            raise Exception("Model not fitted yet")
        return self.m*X + self.b
    
    def get_params(self):
        if self.m is None or self.b is None:
            raise Exception("Model not fitted yet")
        return np.array([self.m,self.b])
    
    def plot_regression_line(self, X, y):
        plt.scatter(X, y)
        y_pred = self.predict(X)
        print(f"The solution is y = {self.m}X + {self.b}")
        plt.plot(X, y_pred, color='red')
        plt.show()
        
    def fit(self, X, y):
        n = len(X)
        self.m = (n * np.sum(X*y) - np.sum(X) * np.sum(y)) / (n*np.sum(X**2) - np.sum(X)**2)
        self.b = (np.sum(y) - self.m * np.sum(X)) / n



