import numpy as np
import os
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
import random

tfb = tfp.bijectors
tfd = tfp.distributions

def make_train_X(train_pri) :
    train_X = np.zeros((len(train_pri),1)) 
    for j in range(len(train_pri)) :
        temp = train_pri.iloc[j].astype(str)
        temp2 = temp[0].split(":")
        train_X[(j,0)] = 60*float(temp2[0]) + float(temp2[1])
    
    return train_X

def HR_dataloader(nameprefix="train"):
    path = os.path.join("data","physionet_split",nameprefix)
    dirs = os.listdir(path)
    train_X = []
    train_Y = []
    train_Z = []

    for item in dirs : 
        # Load one data
        my_tupule = ("data","physionet_split", nameprefix,item)
        filepaths  =  "/".join(my_tupule)
        data_pri = pd.read_csv(filepaths, sep=",")
        data_HR_pri = data_pri[data_pri["Parameter"]=='HR']
        data_HR = data_HR_pri[["Time","Value"]]
        #Extract the value of HR data
        train_Y_pri_pri = data_HR[["Value"]]
        train_Y_pri = pd.DataFrame.to_numpy(train_Y_pri_pri)
        #Extract the time of HR data
        train_X_pri_pri = data_HR[["Time"]]
        
        train_X_pri = make_train_X(train_X_pri_pri)

        my_pred_range = range(0,len(train_X_pri),1)
        train = np.zeros((len(train_X_pri),1))
        target = np.zeros((len(train_X_pri),1))

        for j in my_pred_range :
            train[j,:] = train_X_pri[j].T/500
            target[j,:] = train_Y_pri[j]/50
            
        train_X.append(train)
        train_Y.append(target)
        train_Z.append(int(item[:-4]))

    train_global_pre = []
    target_global_pre = []
    
    pred_range_train_global = range(0,len(train_X),1)

    for j in pred_range_train_global :
        train_temp = train_X[j][:][:].reshape(1,-1,1)
        target_temp = train_Y[j].reshape(1,-1,1)
        
        train_global_pre.append(train_temp)
        target_global_pre.append(target_temp)

    train_global = tf.ragged.constant(train_global_pre)
    target_global = tf.ragged.constant(target_global_pre)

    # make as tensor
    train_tensor = tf.data.Dataset.from_tensor_slices((train_global,target_global,train_Z))

    return train_tensor

def HR_dataloader2(nameprefix="Whole"):
    path = os.path.join("data","physionet_split",nameprefix)
    dirs = os.listdir(path)
    train_X = []
    train_Y = []
    train_Z = []

    val_X = []
    val_Y = []
    val_Z = []

    shuffle_whole = np.arange(865)
    random.shuffle(shuffle_whole)
    train_index = shuffle_whole[0:606] 
    val_index = shuffle_whole[606:606+86] 
    iter = 0

    for item in dirs: 
        if iter in train_index:
            # Load one data
            my_tupule = ("data","physionet_split", nameprefix,item)
            filepaths  =  "/".join(my_tupule)
            data_pri = pd.read_csv(filepaths, sep=",")
            data_HR_pri = data_pri[data_pri["Parameter"]=='HR']
            data_HR = data_HR_pri[["Time","Value"]]
            #Extract the value of HR data
            train_Y_pri_pri = data_HR[["Value"]]
            train_Y_pri = pd.DataFrame.to_numpy(train_Y_pri_pri)
            #Extract the time of HR data
            train_X_pri_pri = data_HR[["Time"]]
            
            train_X_pri = make_train_X(train_X_pri_pri)

            my_pred_range = range(0,len(train_X_pri),1)
            train = np.zeros((len(train_X_pri),1))
            target = np.zeros((len(train_X_pri),1))

            for j in my_pred_range :
                train[j,:] = train_X_pri[j].T/500
                target[j,:] = train_Y_pri[j]/50
                
            train_X.append(train)
            train_Y.append(target)
            train_Z.append(int(item[:-4]))

        elif iter in val_index:
            # Load one data
            my_tupule = ("data","physionet_split", nameprefix,item)
            filepaths  =  "/".join(my_tupule)
            data_pri = pd.read_csv(filepaths, sep=",")
            data_HR_pri = data_pri[data_pri["Parameter"]=='HR']
            data_HR = data_HR_pri[["Time","Value"]]
            #Extract the value of HR data
            train_Y_pri_pri = data_HR[["Value"]]
            train_Y_pri = pd.DataFrame.to_numpy(train_Y_pri_pri)
            #Extract the time of HR data
            train_X_pri_pri = data_HR[["Time"]]
            
            train_X_pri = make_train_X(train_X_pri_pri)

            my_pred_range = range(0,len(train_X_pri),1)
            train = np.zeros((len(train_X_pri),1))
            target = np.zeros((len(train_X_pri),1))

            for j in my_pred_range :
                train[j,:] = train_X_pri[j].T/500
                target[j,:] = train_Y_pri[j]/50
                
            val_X.append(train)
            val_Y.append(target)
            val_Z.append(int(item[:-4]))
        iter += 1

    train_global_pre = []
    target_global_pre = []
    
    pred_range_train_global = range(0,len(train_X),1)

    for j in pred_range_train_global :
        train_temp = train_X[j][:][:].reshape(1,-1,1)
        target_temp = train_Y[j].reshape(1,-1,1)
        
        train_global_pre.append(train_temp)
        target_global_pre.append(target_temp)

    train_global = tf.ragged.constant(train_global_pre)
    target_global = tf.ragged.constant(target_global_pre)

    # make as tensor
    train_tensor = tf.data.Dataset.from_tensor_slices((train_global,target_global,train_Z))

    val_train_global_pre = []
    val_target_global_pre = []
    
    pred_range_val_global = range(0,len(val_X),1)

    for j in pred_range_val_global :
        val_temp = val_X[j][:][:].reshape(1,-1,1)
        val_temp = val_Y[j].reshape(1,-1,1)
        
        val_train_global_pre.append(val_temp)
        val_target_global_pre.append(val_temp)

    val_train_global = tf.ragged.constant(val_train_global_pre)
    val_target_global = tf.ragged.constant(val_target_global_pre)

    # make as tensor
    val_tensor = tf.data.Dataset.from_tensor_slices((val_train_global,val_target_global,val_Z))

    return train_tensor, val_tensor