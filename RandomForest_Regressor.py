import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
data=pd.read_csv("C:/Users/shiv/Desktop/Position_Salaries.csv")
real_x = data.iloc[ :,1:2].values
real_y = data.iloc[ :,2].values
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=10,random_state=0)
reg.fit(real_x,real_y)
y_pred=reg.predict(real_x)
plt.scatter(real_x,real_y,color="red")
plt.plot(real_x,y_pred,color="blue")
plt.title("RandomForest_regression")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()


