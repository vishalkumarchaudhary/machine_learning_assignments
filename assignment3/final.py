 

import math as m 
import random
import numpy as np 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering 
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import KernelPCA
from sklearn import mixture

def generatesphere( inner , outer ,numdatapoints ):
    x_data = []
    y_data = []
    z_data = []
    count = 0
    while (count < numdatapoints ):
        r = random.uniform(inner ,outer)
        z = random.uniform(-r , r ) 
        xy = m.sqrt(r**2 -z**2)
        theta = random.uniform( 0 , 2 * m.pi )
        x_data = np.append(x_data , xy*m.cos(theta))
        y_data = np.append(y_data , xy*m.sin(theta))
        z_data = np.append(z_data , z )
        count = count +1 
        
    return (x_data ,y_data ,z_data )


(x_data1 , y_data1 ,z_data1) = generatesphere( 9,10 ,500 )
(x_data2 , y_data2 ,z_data2) = generatesphere(19 ,20 ,500 )
(x_data3 , y_data3 ,z_data3) = generatesphere(29 ,30 ,500 )
 

fig = pyplot.figure()
ax = Axes3D(fig)
ax.set_title("generated data points")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.scatter(x_data1, y_data1, z_data1 ,c='red')
ax.scatter(x_data2, y_data2, z_data2 ,c='blue')
ax.scatter(x_data3, y_data3, z_data3 ,c='green')
pyplot.show() 

 
xdatas =  np.append(x_data1 ,[ x_data2 , x_data3 ]) 

ydatas =np.append(y_data1 ,[ y_data2  , y_data3 ] )
zdatas = np.append(z_data1 ,[ z_data2 , z_data3 ])

data = list (zip(xdatas ,ydatas ,zdatas ))

#########################  kmeans clustering ##############################

 

def plotfig(noofclustes  , plabels ,data ,title ,axis) :
    datacluster = []
    for i in range( noofclustes ):
        datacluster.append([i for i in range(axis)] )

    for counter , i in enumerate(plabels,0):
        datacluster[i] = np.vstack((datacluster[i] ,   data[ counter ]   ) )

    datacluster = [datacluster[0][1:],datacluster[1][1:],datacluster[2][1:]]

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    if(axis==3):
        ax.set_zlabel('Z Label')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    c=["red","green","blue"]
    for i in range(axis):
        ax.scatter(  datacluster[i][:,0], datacluster[i][:,1], datacluster[i][:,2] ,c= c[i] )
    
    pyplot.show() 

def generatetruelabels() :
    labels=[] 
    
    for j in range(3) :
        for i in range(500):
                labels = np.append(labels , j)
    return labels 
        
    
def findingaccuracy(truelabels ,predictedlabels):
    count=0
    for i in range(len(truelabels)) :
        if(truelabels[i] == predictedlabels[i]):
            count=count+1 
            
    return count/len(truelabels)
n_clusters = 3
truelabels = generatetruelabels()

def kmeansclustering(data , n_clusters  ) :
    kmeans = KMeans(n_clusters=n_clusters , random_state=0  ).fit(data)
    plabels = kmeans.labels_
    return plabels
   
    
plabels = kmeansclustering(data , 3  )
plotfig(n_clusters , plabels ,data ,"k-means clustering result" ,3)
acc = findingaccuracy(truelabels , plabels )
 

#################### spectral clustering ########################################

def spectralclustering(data , n_clusters ) :
    spectral = SpectralClustering( n_clusters=3 ,affinity="nearest_neighbors" ,random_state=0).fit(data)
    plabels = spectral.labels_
    return plabels
    
    
plabels = spectralclustering(data , 3  ) 
acc = findingaccuracy(truelabels , plabels)
plotfig(n_clusters ,plabels ,data ,"spectral clustering result", 3 )
 

############################################# gmm clustering ############################################################ 



def fit_samples(data,n_cluters  ):
    gmm = GaussianMixture(n_components=n_cluters )
    gmm.fit(data)
    
     
    return gmm.predict(data)

plabels = fit_samples(data,3 )

plotfig(n_clusters , plabels ,data ,"gmm clustering result" , 3)
acc = findingaccuracy( truelabels, plabels )
 
 

#########################################################################################################################

############################# part 2 starts here ###################### 


def plotfig2d(noofclustes  , plabels ,data  ,title_ ) :
    datacluster = []
    for i in range( noofclustes ):
        datacluster.append([1,1] )
    
     
    for counter , i in enumerate(plabels,0):
        datacluster[i] = np.vstack((datacluster[i] ,   data[ counter ]   ) )
     
    datacluster = [datacluster[0][1:],datacluster[1][1:],datacluster[2][1:]]

    pyplot.figure()
    pyplot.title(title_)
    
    pyplot.xlabel('X Label')
    pyplot.ylabel('Y Label')

    c=["red","green","blue" ]
    for i in range(noofclustes):
        pyplot.scatter(  datacluster[i][:,0], datacluster[i][:,1]  , c=c[i] )
    
    pyplot.show() 
    
def plotfig1d(noofclustes  , plabels ,data  ,title_ ) :
    datacluster = []
    for i in range( noofclustes ):
        datacluster.append([1] )
    
     
    for counter , i in enumerate(plabels,0):
        datacluster[i] = np.vstack((datacluster[i] ,   data[ counter ]   ) )
     
    datacluster = [datacluster[0][1:],datacluster[1][1:],datacluster[2][1:]]

    pyplot.figure()
    pyplot.title(title_)
    pyplot.xlabel('X Label')
    pyplot.ylabel('Y Label')

    c=["red","green","blue" ] 
    for i in range(noofclustes):
        pyplot.scatter(  datacluster[i] , [i]*len(datacluster[i])  , c=c[i] )
    
    pyplot.show() 
    
def pcacomponentred(component):
    pca = PCA(n_components=component )
    pca_data = data
    pca_data = pca.fit_transform(pca_data)


    plabels =kmeansclustering(data=pca_data,n_clusters=3 )
    plotfig2d(3 , plabels ,pca_data ," pca k-means clustering " )
   


    plabels =spectralclustering(data=pca_data,n_clusters=3 )
    plotfig2d(3 , plabels ,pca_data ,"pca Spectral  clustering " )
   

    plabels =fit_samples(pca_data, 3 )
    plotfig2d(3 , plabels ,pca_data ,"pca gmm clustering " )
   
    
pcacomponentred(2)

def pcacomponentred1(component):
    pca = PCA(n_components=component )
    pca_data = data
    pca_data = pca.fit_transform(pca_data)


    plabels =kmeansclustering(data=pca_data,n_clusters=3 )
    plotfig1d(3 , plabels ,pca_data ,"pca k-means clustering " )
    


    plabels =spectralclustering(data=pca_data,n_clusters=3 )
    plotfig1d(3 , plabels ,pca_data ,"pca spectral clustering " )
    

    plabels =fit_samples(pca_data, 3 )
    plotfig1d(3 , plabels ,pca_data ,"pca gmm clustering " )
    
    
pcacomponentred1(1)



##########   KPCA 


def pkernel(X, Y):
    return (X.dot(Y.T)+1)**2/1500

X = np.array(data )
 
gram = pkernel(X, X)
print(np.shape(gram))
def kpca(components):
    kernelpca = PCA( n_components = components )
    kpca_data = kernelpca.fit_transform( pkernel(X,X) )
    
    plabels = kmeansclustering(data=kpca_data , n_clusters= 3)
    
    
    plotfig2d(3  , plabels ,kpca_data  ,"kpca k-means clustering ")

    plabels =spectralclustering(data=kpca_data,n_clusters=3 )
    plotfig2d(3  , plabels ,kpca_data  ," kpca spectral clustering " )
    
    
    
    
    plabels =fit_samples(kpca_data, 3 )
    plotfig2d(3  , plabels ,kpca_data  ," Kpca gmm clustering  " )
    

kpca(2)

def kpca1(components):
    kernelpca = KernelPCA( kernel="precomputed" , n_components = components )
    kpca_data = kernelpca.fit_transform( gram , X )
 
    plabels = kmeansclustering(data=kpca_data , n_clusters= 3)
    plotfig1d(3  , plabels ,kpca_data  ,"kpca k-means clustering ")
    
    
    
    plabels =spectralclustering(data=kpca_data,n_clusters=3 )
    plotfig1d(3  , plabels ,kpca_data  ," kpca spectral clustering " )
    
    
    
    
    plabels =fit_samples(kpca_data, 3 )
    plotfig1d(3  , plabels ,kpca_data  ," Kpca gmm clustering  ")
    
        
kpca1(1)



print("*******************************part 3**********************")

 

x_data = np.genfromtxt("cancer.data",delimiter=',')

np.random.shuffle(x_data)

x_feature = x_data[:,1:10]
 

labels = x_data[:,10]
train_data = x_feature[:int(.6*len(x_data))]
labeltrain = labels[:int(.6*len(x_data))]
test_data = x_feature[int(.6*len(x_data)): ]
labeltest = labels[int(.6*len(x_data)):]

x = [] 
y = []
accu = []
for depth in range(8 ,20) :
	for num_tree in range(9,20) :
		x = np.append(x , depth) 
		y = np.append(y , num_tree)

		rf = RandomForestClassifier( max_depth = depth ,n_estimators = num_tree)
		rf.fit(train_data , labeltrain )
		accu = np.append(accu , rf.score( test_data,labeltest ) )

print(max(accu))
i = np.argmax(accu)

print("optimum depth ,number of trees is" , x[i] , y[i])

forest = RandomForestClassifier(n_estimators=int(y[i]), max_depth=int(x[i]))
forest.fit(train_data, labeltrain)
print(forest.score(test_data, labeltest))

# comparing with its decision tree which has maximum accuracy

labeltest = (np.array(labeltest)/2-1)

estimatorScores = [] 
for estimator in forest.estimators_ :
    estimatorScores.append(estimator.score(test_data, labeltest ))

print(estimatorScores) 

