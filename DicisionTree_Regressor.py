import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
data=pd.read_csv('C:/Users/shiv/Desktop/Position_Salaries.csv')
real_x=data.iloc[ :,1:2].values
real_y=data.iloc[ :,2].values
from sklearn.tree import DecisionTreeRegressor
dtr =  DecisionTreeRegressor(random_state=0)
dtr.fit(real_x,real_y)
y_pred=dtr.predict(real_x)
plt.scatter(real_x,real_y,color="red")
plt.plot(real_x,y_pred,color="blue")
plt.title("Decision_tree_regression")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

