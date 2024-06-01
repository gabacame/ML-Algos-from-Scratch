import pandas as pd
from LinReg import LinReg 
from Split import Split

data = pd.read_csv('Lin Reg/lin_data.csv')

X = data['X']
y = data['y']

Model = LinReg()

X_train, y_train, X_predict, y_predict = Split(X, y, 0.72)

Model.fit(X_train, y_train)
print(Model.get_params())
print(Model.Score(X_predict,y_predict))
Model.plot_regression_line(X,y)


