from svmutil import *
from svm import *
import numpy as np
import csv 


#this function takes the label file and according to label it labels it +1 and other labels to be -1
#and outputs the file compatibe with libsvm 
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
    
make_file_compatible_libsvm("USPSTrain.csv" ,"USPSTrainLabel.csv" ,"output.csv")       
    
'''***************************************************************************************************'''
def getKernelSVMSolution(xtr ,ytr,c,lamda ,xts) :
    
    make_file_compatible_libsvm(xtr ,ytr ,"output.csv")   
    y, x = svm_read_problem('output.csv')
    
    #param =  svm_parameter('-s 0 -t 2 -c '+str(c) +' -g '+str(lamda))
    param = svm_parameter('-s 0 -t 2 -c '+str(float(c)) +' -g '+str(float(3/lamda)))
    prob  = svm_problem(y, x)
    
    m = svm_train(prob,param)
    #m = svm_train(y , x ,' -c '+ str(c))
    #opening file for creating fake label file o.csv
    xts1 = np.genfromtxt(xts ,delimiter='\n' )
 
    #making temporary file for labels of test
    x_out = open("o.csv","w")
    for i in range(len(xts1)):
        x_out.write("1 "+'\n')
    x_out.close()
    
    #making the test data compatible with lib svm
    make_file_compatible_libsvm(xts ,"o.csv" ,"output1.csv") 
    yts, Xts = svm_read_problem('output1.csv')
    
    #predictiong the value
    p_label, p_acc, p_val = svm_predict([0]*len(Xts),Xts, m )
    print(p_label)
    return p_label

'''*************************************************************************************************'''

#this function returns the predicted label for the test data 

getKernelSVMSolution("USPSTrain.csv",'USPSTrainLabel.csv',1,100,'USPSTest.csv')




