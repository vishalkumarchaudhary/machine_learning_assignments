import numpy as np

#reading the training data and calculating median of square distace b/w them
x = np.genfromtxt('USPSTrain.csv' ,delimiter=',' ,autostrip=True)

s=0
for i in range(len(x) ):
    for j in range(i+1 ,(len(x))):
        l = np.linalg.norm(x[i]-x[j])
        s+=l**2


lamda = 3/(s/len(x))
print(lamda)