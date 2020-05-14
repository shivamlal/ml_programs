import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
x_data= np.linspace(0.0,10.0,100000)
noise = np.random.rand(len(x_data))
y_true = (0.5*x_data)+5+noise
x_df=pd.DataFrame(x_data, columns=['x_data'])
y_df=pd.DataFrame(y_true,columns=['y'])
my_data=pd.concat([x_df,y_df],axis=1)
batchsize=15
m=tf.Variable(0.5)
b=tf.Variable(1.0)
xph=tf.placeholder(tf.float32,[batchsize])
yph=tf.placeholder(tf.float32,[batchsize])
y_model=m*xph+b
error=tf.reduce_sum(tf.square(yph-y_model))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(error)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches=100000
    for i in range(batches):
        rand_ind=np.random.randint(len(x_data),size=batchsize)
        feed={xph:x_data[rand_ind],yph:y_true[rand_ind]}
        sess.run(train,feed_dict=feed)
    model_m,model_b=sess.run([m,b])
y_hat=x_data*model_m+model_b
my_data.sample(n=250).plot(kind='scatter',x='x_data',y='y')
plt.plot(x_data,y_hat,'r')
plt.show()
