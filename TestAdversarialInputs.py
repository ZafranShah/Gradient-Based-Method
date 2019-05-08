# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:44:45 2019

@author: zhshah
"""

from __future__ import division
import pickle ,gzip
import matplotlib.pyplot as plt


advImagePath= '../adversarialinputs.pkl.gz'
savedModel = 'svm_model.sav'

def LoadAdvInputs(advpath):
    with gzip.open(advpath,'rb') as infile:
        data = pickle.load(infile)
        advImages = data
        return advImages

def LoadTrainedModel(filename):
    SvmModel = pickle.load(open(filename, 'rb'))
    return SvmModel

adversarialImages= LoadAdvInputs(advImagePath)
SvmModel= LoadTrainedModel(savedModel)
############# Predicting the Adversarial Images ####################
adv_pred=SvmModel.predict(adversarialImages)


plt.imshow(adversarialImages[0].reshape(28,28), cmap="gray")

    