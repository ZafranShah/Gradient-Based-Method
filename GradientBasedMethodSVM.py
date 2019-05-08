# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:18:39 2019

@author: zhshah
"""

from __future__ import division
import numpy as np
import pickle ,gzip
from sklearn import svm
import math
from sklearn.metrics import accuracy_score
import heapq


########################Command Line Inputs #########################################

data_Path ='/../mnist.pkl.gz'
savdir= '' 
adversarial_inputs=[]
epsilon=1  
pathadv= 'adversarialinputs.pkl.gz'

#################Adversarial Inputs in SVM#########################
class GradientBaseMethodSVM:
    
    def __init__(self, dataPath, Directory):
        self.path= dataPath
        self.savedir= Directory
    
    def DataExtraction(self):
        if self.path.strip():
            f=gzip.open(self.path,'rb')
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            xTrain, yTrain= train_set
            xTest, yTest=test_set
            valid_x, valid_y= valid_set
            return  xTrain, yTrain, xTest, yTest, valid_x, valid_y
        else:
            print ('Please provide the path of the data')
        

    def SVMClassifier(self, trainx, trainy, testx):
        print ("Support vector Machine starts")
        if len(trainx) is not 0:
            clf=svm.SVC(C=5.0, cache_size=3000, class_weight=None, 
               coef0=0.0, decision_function_shape='ovr', 
               degree=3, gamma=0.05, kernel='rbf', 
               max_iter=-1, probability=True, 
               random_state=None, shrinking=True, 
               tol=0.001, verbose=False)
            clf.fit(trainx, trainy)
            predy=clf.predict(testx)
            return predy,clf 
        else:
            print('Please load the training data to train the model')
    def SavingSVMModel(self, model):
        print ('save the model to disk')
        filename = 'svm_model.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    def GradientBasedMethod(self, input_image, test_y ):
        sv = svmModel.support_vectors_
        nv = svmModel.n_support_
        gamma=svmModel.gamma
        pred_y=svmModel.predict([input_image])
        if pred_y==test_y:
            probab=svmModel.predict_proba([input_image])
            z=list(probab[0])
            P=heapq.nlargest(2,range(len(z)), key=z.__getitem__)
            if P[0]>P[1]:
                pos, neg =P[1], P[0]
            else:
                pos, neg=P[0], P[1]
            start = [sum(nv[:i]) for i in range(len(nv))]
            end = [start[i] + nv[i] for i in range(len(nv))]
            alpha_pos = (svmModel.dual_coef_[neg - 1, start[pos]:end[pos]])
            alpha_neg = (svmModel.dual_coef_[pos, start[neg]:end[neg]])
            sv_pos = sv[start[pos]:end[pos]]
            sv_neg = sv[start[neg]:end[neg]]
            gradKerPos=0
            gradKerNeg=0
            for i in range(len(sv_pos)):
                x=sv_pos[i]-input_image
                xNorm=np.linalg.norm(x)
                gradKerPos= gradKerPos + 2*gamma*alpha_pos[i]*math.exp(-gamma*(xNorm)**2)*x
            for j in range(len(sv_neg)):
                x_neg=sv_neg[j]-input_image
                xNorm_neg=np.linalg.norm(x_neg)
                gradKerNeg= gradKerNeg + 2*gamma*alpha_neg[j]*math.exp(-gamma*(xNorm_neg)**2)*x_neg
            Gradient = gradKerPos + gradKerNeg
            return Gradient
    def SavingAdversarialImages(self, pathAdvInputs, advInputs):
        if pathAdvInputs is not 0:
            with gzip.open(pathAdvInputs, 'wb') as outfile:
                pickle.dump(advInputs,outfile)
                outfile.close()
        else:
            print ('please provide the path and name')
        








##########Data Extraction and Training Support Vector Machine ######################   
obj= GradientBaseMethodSVM(data_Path, savdir)    
train_x, train_y, test_x, test_y, valid_x, valid_y=obj.DataExtraction()   
pred_y, svmModel= obj.SVMClassifier(train_x, train_y, test_x)
print ("Classification is Done")
accuracy= accuracy_score (test_y, pred_y)
print ('The accuracy of the classifier is:',accuracy)
probability=svmModel.predict_proba(test_x)
decision_function_value=svmModel.decision_function(test_x)
print ('Saving model to disk')
obj.SavingSVMModel(svmModel)

####################Generating Adversarial Inputs#######################

for i in range(0, len(test_y)):
    pred=svmModel.predict([test_x[i]])
    if pred == test_y[i]:
        computedGradient=obj.GradientBasedMethod(test_x[i], test_y[i])
        gradientNorm=np.linalg.norm(computedGradient)
        gradientNormMagnitude=computedGradient/gradientNorm
        product=epsilon*gradientNormMagnitude
        adversarialInput= test_x[i]+ product
        adv_pred=svmModel.predict([adversarialInput])
        if adv_pred != test_y[i]:
            adversarial_inputs.append(adversarialInput)
        else:
            pass
    else:
        continue
################### Saving Adversarial Inputs  ########################
obj.SavingAdversarialImages(pathadv,adversarial_inputs ) 
print ('Adversarial Images are saved on the path', pathadv)      
  
    
########################################
print('out of total',len(test_y), 'images the Gradient Based algorithm manage to convert only', len(adversarial_inputs),'images into adversarial images')
print('The adversarial inputs are cultivated')
print ('Completed')
 
