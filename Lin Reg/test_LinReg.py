import pandas as pd
from LinReg import LinReg 
import numpy as np

data = pd.read_csv('Lin Reg/lin_data.csv')

X = data['X']
y = data['y']

Model = LinReg()

def Split(X, y, split):
    X_train = X[:(int(len(X)*split)-1)]
    y_train = y[:(int(len(y)*split)-1)]
    X_predict = X[(int(len(X)*split)-1):]
    y_predict = y[(int(len(y)*split)-1):]
    return X_train, y_train, X_predict, y_predict

def MSE(X_predict, y_predict, pred):
    y_pred = lambda x: pred.predict(x)
    cost = np.array(list(map(y_pred,X_predict)))
    return np.mean((y_predict-cost)**2)

X_train, y_train, X_predict, y_predict = Split(X, y, 0.8)

Model.fit(X_train, y_train)
print(Model.get_params)
print(MSE(X_predict, y_predict, Model))

