
# coding: utf-8

# In[207]:


import numpy as np
from numpy.linalg import inv
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

data = np.genfromtxt("Features_Variant_5.csv" ,delimiter=',' )
 
l =len(data)//5
train1 ,train2,train3,train4,train5 = np.split(data ,[l,2*l,3*l,4*l,])


# In[208]:


def train(data ,sigma , lamb):
    y=data[:,53]
    data =data[:,:53]
    w =np.matmul( inv(np.matmul(data.T ,data)*sigma/2 +np.identity(53)*lamb/2 ),np.matmul(data.T,y)*sigma/2)
    return w
    
    


# In[209]:


def test(data,w):
    y=data[:,53]
    data =data[:,:53]
    prediction = np.matmul(data,w)
    deflection =np.subtract(y,prediction)
    error =0 
    for i in deflection:
        error = error+i*i
    return np.sqrt(error/len(y))
    
    


# In[222]:


def plot(m,l):
    
    error=[0,0,0,0]
    w=train(train1 ,m , l)
    error[0]=test(train5,w)


    x=np.vstack((train1,train2))
    w=train(x ,m , l)
    error[1]=test(train5,w)

    x=(np.vstack((x,train3)))
    w=train(x ,m , l)
    error[2]=test(train5,w)

    x=(np.vstack((x,train4)))
    w=train(x ,m , l)
    error[3]=test(train5,w)

    le=len(train5)
    x =[le,2*le,3*le,4*le]
    plt.plot(x ,error,'*')
    plt.title("mu =" +str(m) +" lamda = "+str(l) )
    plt.show() 

plot(10,1000)
plot(1,1000)

plot(1,10000)

plot(10000,1)

plot(10000,10000)

plot(500,500)

plot(500000,500000)

#l =len(data)//5
#train1 ,train2,train3,train4,train5 = np.split(data ,[l,2*l,3*l,4*l,])




