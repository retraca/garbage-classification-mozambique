from data import Data

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Activation, Conv1D, MaxPooling1D,LeakyReLU, Dropout, LSTM, Bidirectional, GRU, BatchNormalization, ELU, Attention, LSTM, Input, UpSampling1D, TimeDistributed, SpatialDropout2D, SpatialDropout1D, concatenate

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity
from scipy import stats



def activate_kde(X_treino,Y_treino,bw):
    '''funcao recebe data treino e parametro de alrgura de banda para bw
    retorna dicionarios com as variaveis como key e as suas densidades como
    values'''
    X_train_0,Y_train_0=X_treino[Y_treino==0,:],Y_treino[Y_treino==0] #variaveis treino X class 0 #varaiveis resposta ==0
    X_train_1,Y_train_1=X_treino[Y_treino==1,:],Y_treino[Y_treino==1] #varaiveis resposta == 1 #variaveis treino X class 1

    feats=[i for i in range(np.shape(X_treino)[1])] #definir lista numero de variveis 
    KDE_train0={} #dicionario com o value kde para cada feautures key cls==0
    KDE_train1={} #dicionario com o value kde para cada feautures key cls==1
    #fazer ciclo para estimar a densidade de cada varivel para ambas as classes
    for feat in feats:
        #achar densidade atraves da KDE para class==0
        kde_t0 = KernelDensity(bw, kernel='gaussian')
        kde_t0.fit(X_train_0[:,[feat]],Y_train_0)
        KDE_train0[feat] = (kde_t0)  
        #achar densidade atraves da KDE para class==1        
        kde_t1 = KernelDensity(bw, kernel='gaussian')
        kde_t1.fit(X_train_1[:,[feat]],Y_train_1)
        KDE_train1[feat] = (kde_t1)  
    return KDE_train0,KDE_train1

def prever(x,dic_train0,dic_train1,priori0,priori1):
    '''recebe uma nova entrada e calcula a probabilidade de pertencer a uma deteerminada
    class através da more likelihood - ou seja encontrar o argumento y que maximiza a probabilidade
    devolve array com a previsao de class para o input x
    '''
    p_feats_0=[]
    p_feats_1=[]
    #iterar sobre as keys dos dic fit_kde e calcular a prob like para cada variavel
    for i in dic_train0.keys():
        pred_0=dic_train0[i].score_samples(x[:,[i]]) #calcualr o log likelihood cls0
        pred_1=dic_train1[i].score_samples(x[:,[i]]) #calcualr o log likelihood cls1
        p_feats_0.append(pred_0)
        p_feats_1.append(pred_1)    
    #somatorio de probabilidades da entrada para cada varaivel das diferentes features acordo cls 0 e 1
    soma_prob0=np.sum(p_feats_0,axis=0)
    soma_prob1=np.sum(p_feats_1,axis=0)  
    #verificar argmax y class 
    calc_0 = soma_prob0+math.log(priori0) #probabilidade de ser class 0
    calc_1 = soma_prob1+math.log(priori1) #prob ser class 1
    previsao=calc_0-calc_1
    previsao=np.where(previsao>=0,0,1)
    return previsao

all_picture_data_path = 'versions/3000nvlr_scale_of_5.csv'
data_all = Data(all_picture_data_path)

print(data_all.get_all_data())

X_train,Y_train,X_test,Y_test= data_all.split_train_and_test()
#print(X_train,Y_train,X_test,Y_test)

#probabilidade priori
priori_0=(np.shape(X_train[Y_train==1,:])[0])/np.shape(X_train)[0] #numero de elementos da respectiva clss / numero total
priori_1=1-priori_0 

kf = StratifiedKFold(n_splits = 5) #divisão será estratificada data será dividida em 5 folds treinará  em 4 e validará em 1
NB_kde_te = [] #erro treino
NB_kde_ve = [] #erro validacao
H=np.arange(0.02,0.6,0.02) #definir array bandwith 
for h in H: #iterar sobrebandwith e verificar errors para cada b
    train_error_nv=0
    valid_error_nv=0 #inicializar erros a 0; será feita méia no final sum error/fold
    for train_id,valid_id in kf.split(Y_train,Y_train): # iterar sobre 5 vezes sobre as diferentes divisoes
        #fit de model NB KDE e calcula o 1-score associado (fracao classificaçoes incorrectas)  
        X_tk,Y_tk,X_vk,Y_vk=X_train[train_id],Y_train[train_id],X_train[valid_id],Y_train[valid_id]
        kde_0,kde_1=activate_kde(X_tk,Y_tk,h)
        y_prev_train=prever(X_tk,kde_0,kde_1,priori_0,priori_1) #prev class training set
        t_e=1-accuracy_score(Y_tk, y_prev_train) #calc err para training set
        #repetir proc para prev dados valid
        y_prev_valid=prever(X_vk,kde_0,kde_1,priori_0,priori_1) #prev class valid set
        t_val=1-accuracy_score(Y_vk, y_prev_valid) #calc err para valid set
        train_error_nv+=t_e
        valid_error_nv+=t_val
    NB_kde_te.append(train_error_nv/5)
    NB_kde_ve.append(round((valid_error_nv/5),5))
 
#print('erro treino', NB_kde_te) 
#print('erro_validação',NB_kde_ve)    
#################################################################################
# fazer plot do H (bandwiths eixo x) e erros no y para verificar evolucao erros com h
plt.figure()
plt.title('Naive Bayes KDE - Training vs Validation Error (bandwiths)')
plt.plot(H,NB_kde_te,'blue',label='training_err') #plot erros treino acordo h
plt.plot(H,NB_kde_ve,'red',label='validation_err') #plot erros valid acordo h
plt.xlabel('bandwidth')
plt.ylabel('error')
plt.legend(loc='lower right',frameon=False)
plt.savefig('NB.png',dpi=250) #gravar com nome requerido 
plt.close()

""" 
mdl = Sequential()
mdl.add(keras.Input(shape=(14,129)))
mdl.add(Flatten())
mdl.add(Dense(512))
mdl.add(Activation("relu"))
mdl.add(BatchNormalization()) 
mdl.add(Dropout(0.25))
mdl.add(Dense(512))
mdl.add(Activation("relu"))
mdl.add(BatchNormalization())
mdl.add(Dropout(0.5))
mdl.add(Dense(5*2000))
mdl.add(Reshape((2000,5)))
mdl.add(Activation(activation="softmax")) """