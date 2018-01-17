#   OneVsOne program
from svmutil import *
from svm import *
import numpy as np
import csv 
from collections import *
import matplotlib.pyplot as plt
from sklearn.metrics import *


def make_file_compatible_libsvm(feat_filename , label_filename,output_file ) :
    
    x = np.genfromtxt(feat_filename ,delimiter=',', autostrip=True)
    y_label = np.genfromtxt(label_filename ,delimiter='\n')
    x_out = open(output_file,"w")
    
    counter = 0
     
    for index,i in enumerate(x , 0):
        
        str_=str(int(y_label[index])) + ' '

        for counter ,value in enumerate(i,1) :
            if(value==0 or value==0.0) :
                continue 
            str_ = str_ + str(counter) +':'+str(value)+' '

        x_out.write(str_+'\n')
    x_out.close()
    



#this function takes the label file and according to label it labels it +1 and other labels to be -1
#and outputs the file compatibe with libsvm 
def one_vs_one_compatible_libsvm(feat_filename , label_filename,output_file ,label1 ,label2) :
    
    x = np.genfromtxt(feat_filename ,delimiter=',', autostrip=True)
    y_label = np.genfromtxt(label_filename ,delimiter='\n')
    x_out = open(output_file,"w")
    
    counter = 0
     
    for index,i in enumerate(x , 0):
        
        label_ = y_label[index]
            
        if(label_==label2):
            str_ = '-1 '
        elif(label_ == label1) :
            str_ = '+1 '
        else:
            continue
        
        for counter ,value in enumerate(i,1) :
            if(value==0.0 ) :
                continue 
            str_ = str_ + str(counter) +':'+str(value)+' '

        x_out.write(str_+'\n')  
    x_out.close()


#creating models for all the given  classes in training set
def predict_one_one(xts , m , labels ):
   
    
    score_  =[]
    for i in range(10):
        score_.append(0)
    
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            p_label = libsvm.svm_predict( m[i][j - i -1], xts)
            if(p_label== 1):
                score_[i]+=1 
            elif(p_label == -1):
                score_[j]+=1
        
    return score_



def one_vs_one_models(xtr ,ytr,c,lamda ) :
    
    #getting the number of labels in the training data
    l = np.genfromtxt(ytr,delimiter='\n')
    labels = set(l)
    #print(labels)
    
    m =defaultdict(list)
    for i in range(len(labels)) :
        for j in range(i+1,len(labels)):
            one_vs_one_compatible_libsvm(xtr,ytr,"output.csv",i,j)
            y, x = svm_read_problem('output.csv')
            model_= svm_train(y , x )
            m[i].append(model_)
             
             
    return m ,labels 



#************************************************************************************ 
#as I have got this lamda 
m ,labels = one_vs_one_models("USPSTrain.csv" ,"USPSTrainLabel.csv" ,100,0.00001222989761795)
make_file_compatible_libsvm("USPSTest.csv","USPSTestLabel.csv","output.csv")  

    
predicted_label=[]
yts ,xts = svm_read_problem("output.csv")

p_label_file = open("p_label_file.csv" ,'w')
for i in range(len(xts)):
    xts_, idx = gen_svm_nodearray(xts[i])
    score = predict_one_one(xts_ ,m ,labels)
    p_label_file.write(str(score.index(max(score)))+'\n')
    predicted_label.append(score.index(max(score)))
    
#************************************************************************************
#---------------------calculating f1 ---------------------------
true_predicted_label = 0
for i in range (len(yts)):
    if(predicted_label[i] == yts[i]):
        true_predicted_label +=1

#print("accuracy => ",true_predicted_label/len(yts))
#***************************************************
true_predicted_label = []
for i in range (len(labels)):
    true_predicted_label.append(0)

true_label = []
for i in range (len(labels)):
    true_label.append(0)

predict_by_classifier = []
for i in range (len(labels)):
    predict_by_classifier.append(0)

for i in range(len(labels)):
    for j in range(len(yts)):
        if(yts[j] == i and predicted_label[j] == i ):
            true_predicted_label[i] +=1
        if(yts[j] == i):
            true_label[i] +=1
        if(predicted_label[j] == i):
            predict_by_classifier[i] +=1

recall_ = sum(true_predicted_label)/sum(true_label)
precision_ = sum(true_predicted_label)/sum(predict_by_classifier)
#calculating f1_score
f1_score = 2*recall_*precision_/(recall_+ precision_)

print("f1_score => ",f1_score)
#----------------------------end of calucating f1 score-------------------



#caculating confusion matrix
y_true = ['0','1','2','3','4','5','6','7','8','9']
y_pred = ['0','1','2','3','4','5','6','7','8','9']
mat = confusion_matrix(yts, predicted_label)
print("confusion_matrix")
print(mat)

 

#plotting the missclassified images 
 
gs = plt.GridSpec(1, 5)
counter = 0
xts = np.genfromtxt("USPSTest.csv",delimiter=',' ,autostrip=True)
for j in range(len(yts)):
    if predicted_label[j] != yts[j] :
        labe = 'p: '+str(predicted_label[j])+' t :'+str(int(yts[j]))
        plt.xlabel(labe)
        
        draw = np.array(xts[j])
        draw = draw.reshape(16,16)
        
        img =plt.subplot(gs[counter])
        img.imshow(draw)
        counter +=1
        if(counter==5):
            labe = 'p: '+str(predicted_label[j])+' t :'+str(int(yts[j]))
            plt.xlabel(labe)
            plt.show()
            gs = plt.GridSpec(1, 5)
            counter=0 

