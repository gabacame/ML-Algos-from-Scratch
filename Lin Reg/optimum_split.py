import pandas as pd
from LinReg import LinReg 
import numpy as np
import matplotlib.pyplot as plt
from utils.Split import Split

data = pd.read_csv('Lin Reg/lin_data.csv')

X = data['X']
y = data['y']

Model = LinReg()

splits = np.round(np.arange(0.1,1,0.01),2)

def Score_split(splits):
    X_train, y_train, X_predict, y_predict = Split(X, y, splits)
    Model.fit(X_train, y_train)
    return Model.Score(X_predict,y_predict)

Scores = list(map(Score_split,splits))

score_of_splits = zip(Scores, splits)

print(max(score_of_splits))
plt.plot(splits, Scores)
plt.show()