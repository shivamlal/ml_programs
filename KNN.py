import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv('C:/Users/shiv/Desktop/Social_Network_Ads.csv')
real_x=data.iloc[ : ,[2,3]].values
real_y=data.iloc[:,4].values
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
train_x=scaler.fit_transform(train_x)
test_x=scaler.fit_transform(test_x)
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=5,p=2)
knc.fit(train_x,train_y)
y_pred=knc.predict(test_x)
from sklearn.metrics import confusion_matrix
con_mx=confusion_matrix(test_y,y_pred) 
print(con_mx)

