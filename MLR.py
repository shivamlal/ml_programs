import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
data = pd.read_csv('C:/Users/shiv/Desktop/TSLA.csv')
data.drop('Date',axis=1,inplace=True)
real_x = data.iloc[:,0:6].values
real_y = data.iloc[ : ,-1].values
from sklearn.model_selection import train_test_split
training_x,testing_x,training_y,testing_y = train_test_split(real_x,real_y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(training_x,training_y)
pred_y = MLR.predict(testing_x)
print(testing_y[3:5])
print(pred_y[3:5])


