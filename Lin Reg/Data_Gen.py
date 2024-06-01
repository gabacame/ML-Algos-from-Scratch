import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

params = [2, 2]
des_range=[0.8, 1]

y_f = lambda x_f: x_f*(params[0]*np.random.uniform(low=des_range[0], high=des_range[1])) + (params[1]*np.random.uniform(low=des_range[0], high=des_range[1]))

X = np.random.uniform(low=0 ,high=10 ,size=100)
y = np.array(list(map(y_f,X)))

data = pd.DataFrame(zip(X,y), columns=['X','y'])

plt.scatter(X,y)
plt.show()

data.to_csv('Lin Reg/lin_data.csv', index=False)

