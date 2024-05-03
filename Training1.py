# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 19:45:56 2022

@author:Sizhe Chen
"""
import os
import string
import torch
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D

#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.layers.normalization import batch_normalization
from keras.layers import BatchNormalization


from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D, Conv2D, MaxPooling2D


import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from utils import getMatrixLabel, Phos1,PhosB, PhosE, getMatrixInput, getMatrixInputh, getMatrixLabelFingerprint, getMatrixLabelh, plot_ROC, getMatrixLabelFingerprint1,getMatrixLabelnlp
from keras.optimizers import adam_v2
from utils import channel_attenstion

import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
#
import csv
import numpy as np
import keras.utils.np_utils as kutils
from keras.optimizers import adam_v2
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.models import Sequential, Model
import numpy as np
import keras.utils.np_utils as kutils
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.PyPro import GetProDes
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence as gps
from propy import GetSubSeq
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.AAComposition import CalculateAAComposition
from propy.AAComposition import CalculateAADipeptideComposition
from propy.AAComposition import GetSpectrumDict
from propy.AAComposition import Getkmers
from sklearn.preprocessing import StandardScaler

train_file_name ='TrainingAMP.csv'  # Training dataset
win1 = 50

X1, T,rawseq, length = getMatrixLabelh(train_file_name, win1)
train_file_name = 'Non-AMPsfilter.csv'  # Test dataset

win1 = 50
X1tt, y_train1,rawseq1, length = getMatrixLabelh(train_file_name, win1)

train_file_name = 'Validation.csv' #Validation dataset
win1 = 50
X_val, y_train111,rawseq116, length = getMatrixLabelh(train_file_name, win1)





X2 = np.load(file="Training_vector.npy")# Descriptor of Training dataset
X2tt = np.load(file="Test_vector.npy")# Descriptor of Test dataset
X2_val = np.load(file="5810_vector.npy")# Descriptor of Validation dataset
q=5810
q1=0
z=0
#43404+16000+300
aaa = np.zeros((43404+z+q1+q, 50, 20))
bbb = np.zeros((43404+z+q1+q, 91, 17))
aaa[:43404] = X1[:]
aaa[43404:43404+z] = X1tt[:z]
aaa[43404+z:43404+z+q1] = X1tt[990:990+q1]
aaa[43404+z+q1:43404+z+q1+q] = X_val[:]

bbb[:43404] = X2[:]
bbb[43404:43404+z] = X2tt[:z]
bbb[43404+z:43404+z+q1] = X2tt[990:990+q1]
bbb[43404+z+q1:43404+z+q1+q] = X2_val[:]



ddd = np.zeros(shape=(43404+z+q1+q, 2))
ddd[:43404,0:2] = T[:]



ddd[43404:43404+z,0:2]=y_train1[:z]
ddd[43404+z:43404+z+q1,0:2]=y_train1[z:z+q1]
ddd[43404+z+q1:43404+z+q1+q,0:2] = y_train111[:]

zzz1=np.zeros((39253-z-q1, 50, 20))
zzz2=np.zeros((39253-z-q1, 91, 17))

zzz1[:990-z]=X1tt[z:990]
zzz1[990-z:]=X1tt[990+q1:]

zzz2[:990-z]=X2tt[z:990]
zzz2[990-z:]=X2tt[990+q1:]

dddz = np.zeros(shape=(39253-z-q1, 2))
dddz[:990-z]=y_train1[z:990]
dddz[990-z:]=y_train1[990+q1:]


import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
###################################################################
raw=[]
for i in range(0,len(rawseq)):
 raw.append(rawseq[i])
for i in range(0,len(rawseq1)):
 raw.append(rawseq1[i])
for i in range(0,len(rawseq116)):
 raw.append(rawseq116[i])

train_file_name = 'NLP.csv'  # Training dataset
win1 = 50

nlp, lengthnlp = getMatrixLabelnlp(train_file_name, win1)

for i in range(0,len(nlp)):
 raw.append(nlp[i])
###################################################################
#Transformer 数据构建
from gensim.models import Word2Vec
#model6=Word2Vec(raw[:],sg=1,vector_size=100, window=50, min_count=1, hs=1,workers=4)
#a=wv["A"]
#a1=model6.wv["A"]
#a2=model6.wv["A"]
#-3.67306210e-02

#model6.wv.save_word2vec_format('word2vec11.bin',binary=True)
#model6.wv.save_word2vec_format('word.bin',binary=True)
#model6.wv.load_word2vec_format('word.bin',binary=True)

from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('word2vec11.bin', binary=True)

#model6.save("model6.model")
#from gensim.models import Word2Vec
#model6 = Word2Vec.load("model6.model")


#from gensim.models import KeyedVectors
#model6 = KeyedVectors.load_word2vec_format('word2vec11.bin', binary=True)


vector_size=100
ccc = np.zeros((len(rawseq), 50, vector_size))
for i in range(0,len(rawseq)):
    t=0
    for AA in rawseq[i]:
        ccc[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseq[i])):
        ccc[i][t][:]=-10**(-100000)
        t=t+1
    #print(i)
    
ccc1 = np.zeros((len(rawseq116), 50, vector_size))
for i in range(0,len(rawseq116)):
    t=0
    for AA in rawseq116[i]:
        ccc1[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseq116[i])):
        ccc1[i][t][:]=-10**(-100000)
        t=t+1
    #print(i)

ccc2 = np.zeros((len(rawseq1), 50, vector_size))
for i in range(0,len(rawseq1)):
    t=0
    for AA in rawseq1[i]:
        ccc2[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseq1[i])):
        ccc2[i][t][:]=-10**(-100000)
        t=t+1
    #print(i)

vector=np.zeros((len(rawseq)+z+q1+q, 50, vector_size))

vector[:43404] = ccc[:]
vector[43404:43404+z] = ccc2[:z]
vector[43404+z:43404+z+q1] = ccc2[990:990+q1]
vector[43404+z+q1:43404+z+q1+q] = ccc1[:]

zzz3=np.zeros((39253-z-q1, 50, vector_size))
zzz3[:990-z]=ccc2[z:990]
zzz3[990-z:]=ccc2[990+q1:]
#del raw, rawseq, rawseq1, rawseq116
#del y_train111,y_train1,X_val,X2tt,X1tt,X1,ccc2,lengthnlp,length

img_dim1 = aaa.shape[1:]

img_dim2 = vector.shape[1:]

img_dim3 = bbb.shape[1:]

img_dim4 = bbb.shape[1:]

img_dim5 = bbb.shape[1:]

img_dim6 = bbb.shape[1:]

init_form = 'RandomUniform'
learning_rate = 0.001#0.001
nb_dense_block =9
nb_layers = 9
nb_filter = 16#16
growth_rate = 16#16
filter_size_block1 = 15#15
filter_size_block2 = 75#75
filter_size_block3 = 15#15
filter_size_block4 = 0
filter_size_block5 = 0
filter_size_block6 = 0
filter_size_ori = 1
dense_number = 16#16
dropout_rate = 0.2#0.2
dropout_dense = 0.2#0.2
weight_decay = 0.000001
nb_batch_size = 16#512
nb_classes = 2
nb_epoch = 17
file_name='TrainingAMP.csv'

model1,parameter = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,growth_rate, 
               filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
               nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)
#10 10 20
#10 10 10
#11 11 11
#20 20 20
#11 11 20
#11 7 20
#20 10 20
#11 7 7
#20 19 20
#
# 模型可视化
print(model1.summary())
plot_model(model1, to_file='DTLDephos.png',
           show_shapes=True, show_layer_names=True)

opt = adam_v2.Adam(learning_rate=learning_rate,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                   
#from tensorflow_addons.optimizers import AdamW
#opt = AdamW(learning_rate=learning_rate, weight_decay=1e-4)


model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

#model1.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
#import random
#times=36
history = model1.fit([aaa[:43404],vector[:43404],bbb[:43404]], ddd[:43404], batch_size=nb_batch_size,validation_data=([aaa[43404:],vector[43404:],bbb[43404:]], ddd[43404:]),epochs=nb_epoch, shuffle=True, verbose=1)
    
   
  
    #else:
  #random_numbers = random.sample(range(0, 43404), int(43404*0.99))
  #t1=aaa[random_numbers];t2=vector[random_numbers];t3=bbb[random_numbers];t4=ddd[random_numbers];
#history = model1.fit([aaa[:49704],vector[:49704],bbb[:49704]], ddd[:49704], batch_size=nb_batch_size,validation_data=([aaa[49704:],vector[49704:],bbb[49704:]], ddd[49704:]),epochs=nb_epoch, shuffle=True, verbose=1)

#####
#2024年1月23日
model1.save_weights('T1.h5', overwrite=True)#


#####


model1.save_weights('AMP_Prediction_transformer1.h5', overwrite=True)#0.9493 38167 887
model1.save_weights('AMP_Prediction_transformer2.h5', overwrite=True)#0.9503 38091 915
model1.save_weights('AMP_Prediction_transformer3.h5', overwrite=True)#0.9587 38185 854
model1.save_weights('AMP_Prediction_transformer4.h5', overwrite=True)#0.9534 38095 924
model1.save_weights('AMP_Prediction_transformer5.h5', overwrite=True)#0.9544 38106 919
model1.save_weights('AMP_Prediction_transformer6.h5', overwrite=True)# 

model1.save_weights('AMP_Prediction_transformer11.h5', overwrite=True)#0.946981 38189 863

model1.load_weights('AMP_Prediction_transformer_times117.h5')
#H
model1.save_weights('AMP_Prediction_1.h5', overwrite=True)
#16 16 16 9
model1.load_weights('AMP_Prediction_transformer_times113.h5')
#AMP_Prediction_transformer_times1
#8 39172 850 0.9440916957159637
#13 38165 881 0.9482548259717399
#15 38142 892 0.948546977568092
#17 38122 904 0.9490291389482436
#23 38123 897 0.9421782328090066

#change the layer to 7
#AMP_Prediction_transformer_times2
#16 38122 904 0.9437752280577738
#19 38166 867 0.9439653791694643
#20 38102 914 0.9431692668314048
#24 38147 899 0.9414151075730125

#G
model1.save_weights('AMP_Prediction_transformer_G_1.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_G_2.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_G_3.h5', overwrite=True)
model1.load_weights('AMP_Prediction_transformer_G_1.h5')
#
#13 
#37843 939 0.9148
#37986 864 0.9096
#38138 850 0.9273

#F
nb_epoch=2
model1.save_weights('AMP_Prediction_transformer_F_1.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_F_2.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_F_3.h5', overwrite=True)
model1.load_weights('AMP_Prediction_transformer_F_3.h5')
#38144 838 0.9224
#38105 886 0.9183
#38062 893 0.9258
#

#E
model1.save_weights('AMP_Prediction_transformer_E_1.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_E_2.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_E_3.h5', overwrite=True)
model1.load_weights('AMP_Prediction_transformer_E_3.h5')
#38135 871 0.9320
#38035 900 0.9137
#38064 883 0.9214

#D
model1.save_weights('AMP_Prediction_transformer_D_1.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_D_2.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_D_3.h5', overwrite=True)
model1.load_weights('AMP_Prediction_transformer_D_3.h5')
#38134 845 0.9058
#38076 823 0.8830
#38074 849 0.9072

#C
model1.save_weights('AMP_Prediction_transformer_C_1.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_C_2.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_C_3.h5', overwrite=True)
model1.load_weights('AMP_Prediction_transformer_C_1.h5')
#37891 876 0.8806
#38038 866 0.8999
#38121 820 0.8977

#B A
model1.save_weights('AMP_Prediction_transformer_B_1.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_B_2.h5', overwrite=True)
model1.save_weights('AMP_Prediction_transformer_B_3.h5', overwrite=True)
model1.load_weights('AMP_Prediction_transformer_B_1.h5')
#38144 705 0.8701
#38077 790 0.8664
#37903 877 0.8725

#A

predictions_p = model1.predict([X1tt,ccc2,X2tt])#Evaluating the effects on Test dataset
#predictions_p = model1.predict([aaa[59704:],vector[59704:],bbb[59704:]])#Evaluating the effects on Test dataset
print(np.sum(predictions_p[990:,0]>0.5))#Print AMPs predictions
print(np.sum(predictions_p[:990,1]>0.5))#Print Non-AMPs predictions
#
labels=y_train1[:,1] 
preds=predictions_p[:,1] #savepath='D://'
fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)  ###
precision,recall,threshold1 = metrics.precision_recall_curve(labels, preds)
roc_auc1 = metrics.auc(fpr1,tpr1)  ###计算auc的值，auc
roc_auc1 = metrics.auc(recall,precision)
plt.figure()




#####################Prediction#####################
train_file_name ='c6.csv'  # Training dataset
win1 = 50

Xc, Tc,rawseqc, lengthc = getMatrixLabelh(train_file_name, win1)


######################################

Xc2 = np.load(file="c6.npy")# Descriptor of Training dataset



cccccc = np.zeros((len(rawseqc), 50, vector_size))
for i in range(0,len(rawseqc)):
    t=0
    for AA in rawseqc[i]:
        cccccc[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseqc[i])):
        cccccc[i][t][:]=-10**(-100000)
        t=t+1

#######################Statistical Part##################################
model1.load_weights('AMP_Prediction_transformer_times113.h5')
#AMP_Prediction_transformer_times1
#8 39172 850 0.9440916957159637
#13 38165 881 0.9482548259717399
#15 38142 892 0.948546977568092
#17 38122 904 0.9490291389482436
#23 38123 897 0.9421782328090066




predictions_p1 = model1.predict([Xc,cccccc,Xc2])#Evaluating the effects on Test dataset
print(np.sum(predictions_p1[:,1]>0.99))#Print Non-AMPs predictions
#np.save(file="cp7_1.npy",arr=predictions_p1)
#predictions_p1=np.load(file="cp6_1.npy")

model1.load_weights('AMP_Prediction_transformer_times115.h5')
predictions_p2 = model1.predict([Xc,cccccc,Xc2])#Evaluating the effects on Test dataset
print(np.sum(predictions_p2[:,1]>0.99))#Print Non-AMPs predictions
#np.save(file="cp7_2.npy",arr=predictions_p2)
#predictions_p2=np.load(file="cp6_2.npy")

model1.load_weights('AMP_Prediction_transformer_times117.h5')
predictions_p3 = model1.predict([Xc,cccccc,Xc2])#Evaluating the effects on Test dataset
print(np.sum(predictions_p3[:,1]>0.99))#Print Non-AMPs predictions
#np.save(file="cp7_3.npy",arr=predictions_p3)
#predictions_p3=np.load(file="cp6_3.npy")

#AMP_identifier1
#Please use AMP_identifier1 for prediction
#Path: D:\Paper\Experiments\翻译后修饰预测\新建文件夹\AMPidentifier1\Fast-AMPs-Discovery-Projects-main
#np.save(file="p1.npy",arr=predictions_p)
predictions_p=np.load(file="p1.npy")

#AMP_identifier2
#np.save(file="p.npy",arr=predictions_p)
#predictions_p1=np.load(file="p.npy")




#Finding the satisfactory sequences with potential antimicrobial activities
count=0
for i in range(0,len(rawseqc)):
    if predictions_p1[i,1]>=0.9 and predictions_p2[i,1]>=0.9 and predictions_p3[i,1]>=0.9:
        count=count+1
#Only the sequences match the AMP_identifier1 and AMP_identifier2 will be saved
sequence1=dict()#Calculating the frequencies and their sequences
for i in range(0,len(rawseqc)):
   if predictions_p1[i,1]>=0.9 and predictions_p2[i,1]>=0.9 and predictions_p3[i,1]>=0.9:
     if rawseqc[i] not in sequence1:
        sequence1[rawseqc[i]]=1
     else:
        sequence1[rawseqc[i]]=sequence1[rawseqc[i]]+1

fre=0
for key,value in sequence1.items():
    if value >fre:
        print(value)
        print(key)


#########################Save the sequences in the file######################
outfile = open('cp6.csv', 'w')
a=1;
for key,value in sequence1.items():
   if value >fre and key not in rawseq and key not in rawseq1 and key not in rawseq116:
    outfile.write(key+',')
    outfile.write(str(value)+"\n")

outfile.close()

















###################################################################################
train_file_name ='6.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X1m, Tm,rawseqm, lengthm = getMatrixLabelh(train_file_name, win1)

#np.save(file="MILLKQK.npy",arr=Matr)
X2m=np.load(file="MILLKQK.npy")
vector_size=100
ccccccm = np.zeros((len(rawseqm), 50, vector_size))
for i in range(0,len(rawseqm)):
    t=0
    for AA in rawseqm[i]:
        ccccccm[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseqm[i])):
        ccccccm[i][t][:]=-10**(-100000)
        t=t+1
        
predictions_p = model1.predict([X1m,ccccccm,X2m])
#predictions_p = model1.predict([X1m,X1m,X2m,X2m])








###################Function################################
def fun(positive_position_file_name, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label存储0/1值
    #positive_position_file_name='trainingAMP.csv'
    #window_size=50
    prot = []  #
    pos = []  #
    rawseq = [] #
    all_label = [] #
    all_labela = []
    length=[]
    short_seqs = []
    #half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                #position = int(row[2])
                a=window_size-len(row[0])
                sseq = row[0]#+a*' '
                rawseq.append(sseq)
                b=len(row[0])
                length.append(b)
                #center = sseq[position - 1]
            # 
                all_label.append(int(row[1]))
                #all_labela.append(int(row[2])) 
                #prot.append(row[1])
                #pos.append(row[2])

        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        targetY = all_label
        #targetYa = kutils.to_categorical(all_labela)

        ONE_HOT_SIZE = 20
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0;
        letterDict["C"] = 1;
        letterDict["D"] = 2;
        letterDict["E"] = 3;
        letterDict["F"] = 4;
        letterDict["G"] = 5;
        letterDict["H"] = 6;
        letterDict["I"] = 7;
        letterDict["K"] = 8;
        letterDict["L"] = 9;
        letterDict["M"] = 10;
        letterDict["N"] = 11;
        letterDict["P"] = 12;
        letterDict["Q"] = 13;
        letterDict["R"] = 14;
        letterDict["S"] = 15;
        letterDict["T"] = 16;
        letterDict["V"] = 17;
        letterDict["W"] = 18;
        letterDict["Y"] = 19;
        #letterDict['Z'] = 23

        #
        #
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            #print(seq)
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr, rawseq,targetY,length

###################Classification##########################
file ='cp1.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X1, seq1,T1, length1 = fun(file, win1)
del X1

file ='cp2.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X2, seq2,T2, length2 = fun(file, win1)
del X2


file ='cp3.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X3, seq3,T3, length3 = fun(file, win1)
del X3


file ='cp4.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X4, seq4,T4, length4 = fun(file, win1)
del X4


file ='cp5.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X5, seq5,T5, length5 = fun(file, win1)
del X5

file ='cp6.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X6, seq6,T6, length6 = fun(file, win1)
del X6

file ='cp7.csv'#'TrainingAMP.csv'  # Training dataset
win1 = 50

X7, seq7,T7, length7 = fun(file, win1)
del X7


count1=0
sequence=dict()
for i in range(0,len(seq1)):
    if seq1[i] in seq2[:] and seq1[i] in seq3[:] and seq1[i] in seq4[:] and seq1[i] in seq5[:] and seq1[i] in seq6[:] and seq1[i] in seq7[:]:
        count1=count1+1
        sequence[seq1[i]]=T1[i]
        
#############################################1
count1=np.zeros((len(seq1),1),dtype=int)
sequence=dict()
for i in range(0,len(seq1)):
    if seq1[i] in seq2:
        count1[i,0]=count1[i,0]+1
    if seq1[i] in seq3:
        count1[i,0]=count1[i,0]+1        
    if seq1[i] in seq4:
        count1[i,0]=count1[i,0]+1        
    if seq1[i] in seq5:
        count1[i,0]=count1[i,0]+1
    if seq1[i] in seq6:
        count1[i,0]=count1[i,0]+1
    if seq1[i] in seq7:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3:   
        sequence[seq1[i]]=1
print(1)
######################################2
count1=np.zeros((len(seq2),1),dtype=int)
for i in range(0,len(seq2)):
    if seq2[i] in seq1:
        count1[i,0]=count1[i,0]+1
    if seq2[i] in seq3:
        count1[i,0]=count1[i,0]+1        
    if seq2[i] in seq4:
        count1[i,0]=count1[i,0]+1        
    if seq2[i] in seq5:
        count1[i,0]=count1[i,0]+1
    if seq2[i] in seq6:
        count1[i,0]=count1[i,0]+1
    if seq2[i] in seq7:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3 and seq2[i] not in sequence:   
        sequence[seq2[i]]=1
print(2)
######################################3
count1=np.zeros((len(seq3),1),dtype=int)
for i in range(0,len(seq3)):
    if seq3[i] in seq1:
        count1[i,0]=count1[i,0]+1
    if seq3[i] in seq2:
        count1[i,0]=count1[i,0]+1        
    if seq3[i] in seq4:
        count1[i,0]=count1[i,0]+1        
    if seq3[i] in seq5:
        count1[i,0]=count1[i,0]+1
    if seq3[i] in seq6:
        count1[i,0]=count1[i,0]+1
    if seq3[i] in seq7:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3 and seq3[i] not in sequence:   
        sequence[seq3[i]]=1
print(3)
######################################4
count1=np.zeros((len(seq4),1),dtype=int)
for i in range(0,len(seq4)):
    if seq4[i] in seq1:
        count1[i,0]=count1[i,0]+1
    if seq4[i] in seq2:
        count1[i,0]=count1[i,0]+1        
    if seq4[i] in seq3:
        count1[i,0]=count1[i,0]+1        
    if seq4[i] in seq5:
        count1[i,0]=count1[i,0]+1
    if seq4[i] in seq6:
        count1[i,0]=count1[i,0]+1
    if seq4[i] in seq7:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3 and seq4[i] not in sequence:   
        sequence[seq4[i]]=1

print(4)
######################################5
count1=np.zeros((len(seq5),1),dtype=int)
for i in range(0,len(seq5)):
    if seq5[i] in seq1:
        count1[i,0]=count1[i,0]+1
    if seq5[i] in seq2:
        count1[i,0]=count1[i,0]+1        
    if seq5[i] in seq3:
        count1[i,0]=count1[i,0]+1        
    if seq5[i] in seq4:
        count1[i,0]=count1[i,0]+1
    if seq5[i] in seq6:
        count1[i,0]=count1[i,0]+1
    if seq5[i] in seq7:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3 and seq5[i] not in sequence:   
        sequence[seq5[i]]=1
print(5)
######################################6
count1=np.zeros((len(seq6),1),dtype=int)
for i in range(0,len(seq6)):
    if seq6[i] in seq1:
        count1[i,0]=count1[i,0]+1
    if seq6[i] in seq2:
        count1[i,0]=count1[i,0]+1        
    if seq6[i] in seq3:
        count1[i,0]=count1[i,0]+1        
    if seq6[i] in seq4:
        count1[i,0]=count1[i,0]+1
    if seq6[i] in seq5:
        count1[i,0]=count1[i,0]+1
    if seq6[i] in seq7:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3 and seq6[i] not in sequence:   
        sequence[seq6[i]]=1
print(6)
#######################################################7
count1=np.zeros((len(seq7),1),dtype=int)
for i in range(0,len(seq7)):
    if seq7[i] in seq1:
        count1[i,0]=count1[i,0]+1
    if seq7[i] in seq2:
        count1[i,0]=count1[i,0]+1        
    if seq7[i] in seq3:
        count1[i,0]=count1[i,0]+1        
    if seq7[i] in seq4:
        count1[i,0]=count1[i,0]+1
    if seq7[i] in seq5:
        count1[i,0]=count1[i,0]+1
    if seq7[i] in seq6:
        count1[i,0]=count1[i,0]+1

    if count1[i,0]>=3 and seq7[i] not in sequence:   
        sequence[seq7[i]]=1
print(7)

#####################################################
l=dict()
for s in seq2:
    if s not in l:
        l[s]=1
        



# 6个蛋白序列列表
protein_lists = [
    seq1,
    seq2,
    seq3,
    seq4,
    seq5,
    seq6,
    seq7
]

# 创建一个空集合
unique_sequences = set()

# 遍历每个列表，将序列添加到集合中
for protein_list in protein_lists:
    unique_sequences.update(protein_list)
    
outfile = open('AI发现的抗菌肽.csv', 'w')    
for seq in unique_sequences:
     outfile.write(seq+"\n")
outfile.close()
# 统计集合中的序列总数
total_sequences = len(unique_sequences)

print("总共含有的序列数：", total_sequences)

######################################################
# 定义氨基酸疏水性指数
hydrophobicity_index = {
    'A': 1.8,
    'R': -4.5,
    'N': -3.5,
    'D': -3.5,
    'C': 2.5,
    'Q': -3.5,
    'E': -3.5,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'L': 3.8,
    'K': -3.9,
    'M': 1.9,
    'F': 2.8,
    'P': -1.6,
    'S': -0.8,
    'T': -0.7,
    'W': -0.9,
    'Y': -1.3,
    'V': 4.2
}
number=0
for seq in unique_sequences:
    if (-1<=sum(hydrophobicity_index[aa] for aa in seq)<=1)==1:
        number=number+1
##############################################################
number=0
for seq in sequence:
    if (-1<=sum(hydrophobicity_index[aa] for aa in seq)<=1)==1:
        number=number+1

# 氨基酸pKa值字典
pKa_values = {
    'D': 3.9,
    'E': 4.1,
    'H': 6.5,
    'C': 8.3,
    'Y': 10.1,
    'K': 10.5,
    'R': 12.5
}

# 计算肽序列在pH为7.0时的电荷
from Bio.SeqUtils.ProtParam import ProteinAnalysis
number=0
for seq in unique_sequences:
    a=ProteinAnalysis(seq)
    if a.charge_at_pH(7.0)>0:
        number=number+1


charge = 0
number=0
for seq in unique_sequences:
 for aa in seq:
    if aa in pKa_values:
        pKa = pKa_values[aa]
        if 7.0 > pKa:
            charge += 1
        elif 7.0 < pKa:
            charge -= 1
    else:
        charge=0
 if charge>=0:
     number=number+1
number=0   
for seq in unique_sequences:
    if 25>=len(seq)>=12:
        number=number+1
    


#################################################Calculate the shared sequences in total#######################
def charge(sequence):
    # 氨基酸pKa值字典
 pKa_values = {
    'D': 3.9,
    'E': 4.1,
    'H': 6.5,
    'C': 8.3,
    'Y': 10.1,
    'K': 10.5,
    'R': 12.5
 }

# 计算肽序列在pH为7.0时的电荷
 charge = 0
 number=0
 for aa in sequence:
    if aa in pKa_values:
        pKa = pKa_values[aa]
        if 7.0 > pKa:
            charge += 1
        elif 7.0 < pKa:
            charge -= 1
    else:
        charge=0
 return charge
     
from Bio.SeqUtils.ProtParam import ProteinAnalysis
     
number=0
desire=[]
for seq in sequence:
    a=ProteinAnalysis(seq)
    if (-1<sum(hydrophobicity_index[aa] for aa in seq)<=1)==1:
        if a.charge_at_pH(7.0)>0:
            if 12<=len(seq)<=25:
                number=number+1
                desire.append(seq)




#############################################################
train_file_name ='potential.csv'  # Training dataset
win1 = 50

X1, T,rawseq, length = getMatrixLabelh(train_file_name, win1)
#X2=Matr
#np.save(file="x2.npy",arr=X2)
#X2=np.load(file="x2.npy")
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('word2vec11.bin', binary=True)

#predictions_p1 = model1.predict([X1,X1,X2,X2])

################################################################
from Bio.SeqUtils.ProtParam import ProteinAnalysis
a="KKK"
X = ProteinAnalysis(a)
print("%0.2f" % X.isoelectric_point())
sec_struc = X.secondary_structure_fraction()  # [helix, turn, sheet]
print("%0.2f" % X.aromaticity())
print("%0.2f" % X.molecular_weight())
print("%0.2f" % X.get_amino_acids_percent()['A'])

positive_position_file_name='AI发现的抗菌肽.csv'
positive_position_file_name='AMP大全文章比较.csv'
seq=[]
index=0
with open(positive_position_file_name, 'r') as rf:
      reader = csv.reader(rf)
      for row in reader:
        #position = int(row[2])
        #sseq = row[1]
        seq.append(row[0])
        #center = sseq[position - 1]
            # 可预测的范围包括S、T或Y三个氨基酸，不在此范围的数据不处理
            #if center in sites:
        #.append(int(row[0]))
        index=index+1

ddd = np.zeros(shape=(index, 9))
for i in range(0,index):
        a=ProteinAnalysis(seq[i][:])
        ddd[i][0]=a.isoelectric_point()
        ddd[i][1]=a.aromaticity()
        ddd[i][2]=a.molecular_weight()
        ddd[i][3:6]=a.secondary_structure_fraction()
        ddd[i][6]=a.gravy(scale='KyteDoolitle')#Hydrophobicity
        ddd[i][7]=a.charge_at_pH(7.0)
        ddd[i][8]=a.instability_index()
        
outfile = open('AI发现的抗菌肽Features.csv', 'w')
outfile = open('已经发现的抗菌肽Features.csv', 'w')
i=0    
for seq in seq:
      outfile.write(seq+',')
      outfile.write(str(ddd[i][0])+',')
      outfile.write(str(ddd[i][1])+',')
      outfile.write(str(ddd[i][2])+',')
      outfile.write(str(ddd[i][3])+',')
      outfile.write(str(ddd[i][4])+',')
      outfile.write(str(ddd[i][5])+',')
      outfile.write(str(ddd[i][6])+',')
      outfile.write(str(ddd[i][7])+',')
      outfile.write(str(ddd[i][8])+"\n")
      i=i+1
outfile.close()







##############################################################
import csv

outfile = open('AI发现的抗菌肽Features.csv', 'w', newline='')
csv_writer = csv.writer(outfile)

header = ['Sequence', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8']
csv_writer.writerow(header)

for i, seq in enumerate(unique_sequences):
    row = [seq] + list(ddd[i])
    csv_writer.writerow(row)

outfile.close()
