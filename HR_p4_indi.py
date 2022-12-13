import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

import numpy as np
from tqdm import tqdm 
import logging
import matplotlib.pyplot as plt
import copy
import pickle

from Dataloader_HR import HR_dataloader
from means_keras import Warping_mean, MLP_embed, MLP, GRU, GRU_embed
from cutils import RBF_embed, get_mean_params,get_kernel_params,get_mean_no_embed_params

#train
def train_step(dataloader,
            optimizer_train_mean,
            optimizer_train_ker,
            cluster_pi,
            kernel1,
            mean1,
            observation_noise_variance1,
            kernel2,
            mean2,
            observation_noise_variance2,
            kernel3,
            mean3,
            observation_noise_variance3,
            kernel4,
            mean4,
            observation_noise_variance4):
            
    dataloader = dataloader.shuffle(1024*1024, reshuffle_each_iteration=False)
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []

    #EM-step
    for i, (X, y, Z) in enumerate(dataloader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
        if i == 0:
            mean1(X) #dummy #Initialize
            mean2(X)
            mean3(X)
            mean4(X)

        gp1 = tfd.GaussianProcess(kernel=kernel1,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance1)

        gp2 = tfd.GaussianProcess(kernel=kernel2,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance2)
        
        gp3 = tfd.GaussianProcess(kernel=kernel3,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance3)

        gp4 = tfd.GaussianProcess(kernel=kernel4,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance4)

        #Numerical trick for prob
        loss1_temp = gp1.log_prob(y)
        loss1 = tf.exp(loss1_temp-loss1_temp)
        loss2_temp = gp2.log_prob(y)
        loss2 = tf.exp(loss2_temp-loss1_temp)
        loss3_temp = gp3.log_prob(y)
        loss3 = tf.exp(loss3_temp-loss1_temp)
        loss4_temp = gp4.log_prob(y)
        loss4 = tf.exp(loss4_temp-loss1_temp)

        if i == 0:
            prob_r_inter1 = cluster_pi[0]*loss1
            prob_r_inter2 = cluster_pi[1]*loss2
            prob_r_inter3 = cluster_pi[2]*loss3
            prob_r_inter4 = cluster_pi[3]*loss4
            if (prob_r_inter1>=prob_r_inter2)and(prob_r_inter1>=prob_r_inter3)and(prob_r_inter1>=prob_r_inter4):
                cluster1.append(Z)
            elif (prob_r_inter2>=prob_r_inter1)and(prob_r_inter2>=prob_r_inter3)and(prob_r_inter2>=prob_r_inter4):
                cluster2.append(Z)
            elif (prob_r_inter3>=prob_r_inter1)and(prob_r_inter3>=prob_r_inter2)and(prob_r_inter3>=prob_r_inter4):
                cluster3.append(Z)
            else:
                cluster4.append(Z)
            prob_r = tf.convert_to_tensor([prob_r_inter1,prob_r_inter2,prob_r_inter3,prob_r_inter4])
            prob_r = tf.reshape(prob_r,[1,-1])
        else:
            prob_r_inter1 = cluster_pi[0]*loss1
            prob_r_inter2 = cluster_pi[1]*loss2
            prob_r_inter3 = cluster_pi[2]*loss3
            prob_r_inter4 = cluster_pi[3]*loss4
            if (prob_r_inter1>=prob_r_inter2)and(prob_r_inter1>=prob_r_inter3)and(prob_r_inter1>=prob_r_inter4):
                cluster1.append(Z)
            elif (prob_r_inter2>=prob_r_inter1)and(prob_r_inter2>=prob_r_inter3)and(prob_r_inter2>=prob_r_inter4):
                cluster2.append(Z)
            elif (prob_r_inter3>=prob_r_inter1)and(prob_r_inter3>=prob_r_inter2)and(prob_r_inter3>=prob_r_inter4):
                cluster3.append(Z)
            else:
                cluster4.append(Z)
            prob_r_inter = tf.convert_to_tensor([prob_r_inter1,prob_r_inter2,prob_r_inter3,prob_r_inter4])
            prob_r_inter = tf.reshape(prob_r_inter,[1,-1])
            prob_r = tf.concat((prob_r,prob_r_inter),axis=0)
        
    loss = 0.
    prob_r = tf.keras.utils.normalize(prob_r, order=1, axis=1)

    index = 0
    index1 = 0
    index2 = 0
    index3 = 0
    index4 = 0

    for count2 in range(prob_r.shape[0]):
        check_cluster = tf.where(prob_r[count2,:] == tf.math.reduce_max(prob_r[count2,:]))
        index += 1

        if check_cluster[0][0] == 0:
            index1 += 1   
        elif check_cluster[0][0] == 1:
            index2 += 1
        elif check_cluster[0][0] == 2:
            index3 += 1
        else:
            index4 += 1
    
    #M-step
    for i, (X, y, _) in enumerate(dataloader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
                    
        gp1 = tfd.GaussianProcess(kernel=kernel1,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance1)

        gp2 = tfd.GaussianProcess(kernel=kernel2,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance2)
        
        gp3 = tfd.GaussianProcess(kernel=kernel3,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance3)
        
        gp4 = tfd.GaussianProcess(kernel=kernel4,
                                index_points=X,
                                observation_noise_variance=observation_noise_variance4)
        
        #For one step, update global->individual parameters simultaneously 
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(get_mean_params(gp1)+get_mean_params(gp2)+get_mean_params(gp3)+get_mean_params(gp4))                
            loss1 = gp1.log_prob(y)
            loss2 = gp2.log_prob(y)
            loss3 = gp3.log_prob(y)
            loss4 = gp4.log_prob(y)
            loss_step = -((prob_r[i,0] * (tf.math.log(cluster_pi[0])+loss1) 
                    + prob_r[i,1] * (tf.math.log(cluster_pi[1])+loss2)
                    + prob_r[i,2] * (tf.math.log(cluster_pi[2])+loss3)
                    + prob_r[i,3] * (tf.math.log(cluster_pi[3])+loss4)
                    ))
        grads = tape.gradient(loss_step, get_mean_params(gp1)+get_mean_params(gp2)+get_mean_params(gp3)+get_mean_params(gp4))
        optimizer_train_mean.apply_gradients(zip(grads,get_mean_params(gp1)+get_mean_params(gp2)+get_mean_params(gp3)+get_mean_params(gp4)))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(get_kernel_params(gp1)+get_kernel_params(gp2)+get_kernel_params(gp3)+get_kernel_params(gp4))                
            loss1 = gp1.log_prob(y)
            loss2 = gp2.log_prob(y)
            loss3 = gp3.log_prob(y)
            loss4 = gp4.log_prob(y)
            loss_step = -((prob_r[i,0] * (tf.math.log(cluster_pi[0])+loss1) 
                    + prob_r[i,1] * (tf.math.log(cluster_pi[1])+loss2)
                    + prob_r[i,2] * (tf.math.log(cluster_pi[2])+loss3)
                    + prob_r[i,3] * (tf.math.log(cluster_pi[3])+loss4)
                    ))
        grads = tape.gradient(loss_step, get_kernel_params(gp1)+get_kernel_params(gp2)+get_kernel_params(gp3)+get_kernel_params(gp4))
        optimizer_train_ker.apply_gradients(zip(grads,get_kernel_params(gp1)+get_kernel_params(gp2)+get_kernel_params(gp3)+get_kernel_params(gp4)))

        loss += loss_step
    loss /= (i+1)

    cluster_pi = tf.reduce_sum(prob_r,axis=0) / len(prob_r)
    print([index,index1,index2,index3,index4])  

    return loss,cluster_pi,cluster1,cluster2,cluster3,cluster4

#validation
def valid_step(data_loader,
            n_adapt,
            lr_val,
            ker_val1,
            mean_val1,
            noise_val1,
            ker_val2,
            mean_val2,
            noise_val2,
            ker_val3,
            mean_val3,
            noise_val3,
            ker_val4,
            mean_val4,
            noise_val4):
    loss = 0.
    for i, (X, y, _) in enumerate(data_loader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
        if i == 0:
            mean_val1(X) #dummy
            mean_val2(X) #dummy
            mean_val3(X) #dummy
            mean_val4(X) #dummy

        train_X, train_y = X[:-1, :], y[:-1]
        test_X, test_y = X[-1:, :], y[-1:]
        # model prediction
        gp_new1 = tfd.GaussianProcess(kernel=ker_val1,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val1)

        gp_new2 = tfd.GaussianProcess(kernel=ker_val2,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val2)
        
        gp_new3 = tfd.GaussianProcess(kernel=ker_val3,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val3)
        
        gp_new4 = tfd.GaussianProcess(kernel=ker_val4,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val4)

        optimizer_gp_val1 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val2 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val3 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val4 = tf.optimizers.Adam(learning_rate=lr_val)

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new1))                
                loss_step1 = -gp_new1.log_prob(train_y)
            grads = tape.gradient(loss_step1, get_kernel_params(gp_new1))
            optimizer_gp_val1.apply_gradients(zip(grads,get_kernel_params(gp_new1)))

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new2))                
                loss_step2 = -gp_new2.log_prob(train_y)
            grads = tape.gradient(loss_step2, get_kernel_params(gp_new2))
            optimizer_gp_val2.apply_gradients(zip(grads,get_kernel_params(gp_new2)))

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new3))                
                loss_step3 = -gp_new3.log_prob(train_y)
            grads = tape.gradient(loss_step3, get_kernel_params(gp_new3))
            optimizer_gp_val3.apply_gradients(zip(grads,get_kernel_params(gp_new3)))

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new4))                
                loss_step4 = -gp_new4.log_prob(train_y)
            grads = tape.gradient(loss_step4, get_kernel_params(gp_new4))
            optimizer_gp_val4.apply_gradients(zip(grads,get_kernel_params(gp_new4)))

        if (loss_step1<=loss_step2)and(loss_step1<=loss_step3)and(loss_step1<=loss_step4):
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val1,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val1,
                    predictive_noise_variance=0.)
        elif (loss_step2<=loss_step1)and(loss_step2<=loss_step3)and(loss_step2<=loss_step4):
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val2,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val2,
                    predictive_noise_variance=0.)
        elif (loss_step3<=loss_step1)and(loss_step3<=loss_step2)and(loss_step3<=loss_step4):
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val3,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val3,
                    predictive_noise_variance=0.)
        else:
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val4,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val4,
                    predictive_noise_variance=0.)

        pred = gprm.mean()
        pred = pred[-1:]
        loss += (pred-test_y[0])**2
    loss /= (i+1)
    loss = tf.math.sqrt(loss)
    return loss

#train and valid
def train_and_valid(dataloader_train,
                    dataloader_val,
                    n_epoch,
                    n_adapt,
                    optimizer_train_mean,
                    optimizer_train_ker,
                    cluster_pi,
                    kernel1,
                    mean1,
                    observation_noise_variance1,
                    kernel2,
                    mean2,
                    observation_noise_variance2,
                    kernel3,
                    mean3,
                    observation_noise_variance3,
                    kernel4,
                    mean4,
                    observation_noise_variance4,
                    lr_val,
                    K):

    best_val_err = float('inf')
    best_ker1 = None
    best_mean1 = None
    best_noise1 = None
    best_ker2 = None
    best_mean2 = None
    best_noise2 = None
    best_ker3 = None
    best_mean3 = None
    best_noise3 = None
    best_ker4 = None
    best_mean4 = None
    best_noise4 = None

    best_cluster1 = None
    best_cluster2 = None
    best_cluster3 = None
    best_cluster4 = None

    early_count = 0

    with tqdm(total=n_epoch) as t:
        for i in range(n_epoch):
            loss,cluster_pi,cluster1,cluster2,cluster3,cluster4=train_step(dataloader_train,
                                                                            optimizer_train_mean,
                                                                            optimizer_train_ker,
                                                                            cluster_pi,
                                                                            kernel1,
                                                                            mean1,
                                                                            observation_noise_variance1,
                                                                            kernel2,
                                                                            mean2,
                                                                            observation_noise_variance2,
                                                                            kernel3,
                                                                            mean3,
                                                                            observation_noise_variance3,
                                                                            kernel4,
                                                                            mean4,
                                                                            observation_noise_variance4)

            ker_val1 = copy.deepcopy(kernel1)
            mean_val1 = copy.deepcopy(mean1)
            noise_val1 = copy.deepcopy(observation_noise_variance1)
            ker_val2 = copy.deepcopy(kernel2)
            mean_val2 = copy.deepcopy(mean2)
            noise_val2 = copy.deepcopy(observation_noise_variance2)
            ker_val3 = copy.deepcopy(kernel3)
            mean_val3 = copy.deepcopy(mean3)
            noise_val3 = copy.deepcopy(observation_noise_variance3)
            ker_val4 = copy.deepcopy(kernel4)
            mean_val4 = copy.deepcopy(mean4)
            noise_val4 = copy.deepcopy(observation_noise_variance4)

            error_val = valid_step(dataloader_val,n_adapt,lr_val,ker_val1,mean_val1,noise_val1,ker_val2,mean_val2,noise_val2,ker_val3,mean_val3,noise_val3,ker_val4,mean_val4,noise_val4)
            is_best = error_val <= best_val_err

            if i>=5:
                if is_best:
                    best_val_err = error_val
                    best_ker1 = copy.deepcopy(kernel1)
                    best_mean1 = copy.deepcopy(mean1)
                    best_noise1 = copy.deepcopy(observation_noise_variance1)
                    best_ker2 = copy.deepcopy(kernel2)
                    best_mean2 = copy.deepcopy(mean2)
                    best_noise2 = copy.deepcopy(observation_noise_variance2)
                    best_ker3 = copy.deepcopy(kernel3)
                    best_mean3 = copy.deepcopy(mean3)
                    best_noise3 = copy.deepcopy(observation_noise_variance3)
                    best_ker4 = copy.deepcopy(kernel4)
                    best_mean4 = copy.deepcopy(mean4)
                    best_noise4 = copy.deepcopy(observation_noise_variance4)

                    best_cluster1 = copy.deepcopy(cluster1)
                    best_cluster2 = copy.deepcopy(cluster2)   
                    best_cluster3 = copy.deepcopy(cluster3)   
                    best_cluster4 = copy.deepcopy(cluster4)   

                    early_count = 0              
                    logging.info("Found new best error at {}".format(i))
                else:
                    early_count += 1

            if i<5:
                cluster_pi = tf.ones(shape=[K,],dtype=tf.float64)/K

            t.set_postfix(loss_and_val_err='{} and {}'.format(loss, error_val))
            print(error_val)
            print('\n')
            t.update()
            if (early_count>5)and(i>=5):
                break

    return best_ker1,best_mean1,best_noise1,best_ker2,best_mean2,best_noise2,best_ker3,best_mean3,best_noise3,best_ker4,best_mean4,best_noise4,best_cluster1,best_cluster2,best_cluster3,best_cluster4  

#Sequential Evaluation
def evaluate(data_loader,n_adapt,lr_val,ker_val1,mean_val1,noise_val1,ker_val2,mean_val2,noise_val2,ker_val3,mean_val3,noise_val3,ker_val4,mean_val4,noise_val4):
    
    rmse3 = 0.
    count = 0

    my_cluster1=[]
    my_cluster2=[]
    my_cluster3=[]
    my_cluster4=[]

    for (X, y, Z) in tqdm(data_loader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
        if count == 0:
            mean_val1(X) #dummy
            mean_val2(X) #dummy
            mean_val3(X) #dummy
            mean_val4(X) #dummy

        train_X, train_y = X[:-1, :], y[:-1]
        test_X, test_y = X[-1:, :], y[-1:]

        # model prediction
        gp_new1 = tfd.GaussianProcess(kernel=ker_val1,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val1)

        gp_new2 = tfd.GaussianProcess(kernel=ker_val2,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val2)

        gp_new3 = tfd.GaussianProcess(kernel=ker_val3,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val3)

        gp_new4 = tfd.GaussianProcess(kernel=ker_val4,
                                    index_points=train_X,
                                    observation_noise_variance=noise_val4)

        optimizer_gp_val1 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val2 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val3 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val4 = tf.optimizers.Adam(learning_rate=lr_val)

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new1))                
                loss_step1 = -gp_new1.log_prob(train_y)
            grads = tape.gradient(loss_step1, get_kernel_params(gp_new1))
            optimizer_gp_val1.apply_gradients(zip(grads,get_kernel_params(gp_new1)))

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new2))                
                loss_step2 = -gp_new2.log_prob(train_y)
            grads = tape.gradient(loss_step2, get_kernel_params(gp_new2))
            optimizer_gp_val2.apply_gradients(zip(grads,get_kernel_params(gp_new2)))

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new3))                
                loss_step3 = -gp_new3.log_prob(train_y)
            grads = tape.gradient(loss_step3, get_kernel_params(gp_new3))
            optimizer_gp_val3.apply_gradients(zip(grads,get_kernel_params(gp_new3)))

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(get_kernel_params(gp_new4))                
                loss_step4 = -gp_new4.log_prob(train_y)
            grads = tape.gradient(loss_step4, get_kernel_params(gp_new4))
            optimizer_gp_val4.apply_gradients(zip(grads,get_kernel_params(gp_new4)))

        t = -1
        if (loss_step1<=loss_step2)and(loss_step1<=loss_step3)and(loss_step1<=loss_step4):
            my_cluster1.append(Z)
            train_X, train_y = X[0:t, :], y[0:t]
            test_X, test_y = X[t:, :], y[t:]
            
            gp_new1 = tfd.GaussianProcess(kernel=ker_val1,
                            index_points=train_X,
                            observation_noise_variance=noise_val1)
            
            optimizer_gp_val1 = tf.optimizers.Adam(learning_rate=lr_val)

            for _ in range(n_adapt):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(get_kernel_params(gp_new1))                
                    loss_step1 = -gp_new1.log_prob(train_y)
                grads = tape.gradient(loss_step1, get_kernel_params(gp_new1))
                optimizer_gp_val1.apply_gradients(zip(grads,get_kernel_params(gp_new1)))

            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val1,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val1,
                    predictive_noise_variance=0.)
            
            pred = gprm.mean()
            rmse3 += (pred[-1]-test_y[0])**2

        elif (loss_step2<=loss_step1)and(loss_step2<=loss_step3)and(loss_step2<=loss_step4):
            my_cluster2.append(Z)
            train_X, train_y = X[0:t, :], y[0:t]
            test_X, test_y = X[t:, :], y[t:]
            
            gp_new2 = tfd.GaussianProcess(kernel=ker_val2,
                            index_points=train_X,
                            observation_noise_variance=noise_val2)
            
            optimizer_gp_val2 = tf.optimizers.Adam(learning_rate=lr_val)
            for _ in range(n_adapt):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(get_kernel_params(gp_new2))                
                    loss_step2 = -gp_new2.log_prob(train_y)
                grads = tape.gradient(loss_step2, get_kernel_params(gp_new2))
                optimizer_gp_val2.apply_gradients(zip(grads,get_kernel_params(gp_new2)))

            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val2,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val2,
                    predictive_noise_variance=0.)
            
            pred = gprm.mean()
            rmse3 += (pred[-1]-test_y[0])**2  

        elif (loss_step3<=loss_step1)and(loss_step3<=loss_step2)and(loss_step3<=loss_step4):
            my_cluster3.append(Z)
            train_X, train_y = X[0:t, :], y[0:t]
            test_X, test_y = X[t:, :], y[t:]

            gp_new3 = tfd.GaussianProcess(kernel=ker_val3,
                            index_points=train_X,
                            observation_noise_variance=noise_val3)
            
            optimizer_gp_val3 = tf.optimizers.Adam(learning_rate=lr_val)
            for _ in range(n_adapt):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(get_kernel_params(gp_new3))                
                    loss_step3 = -gp_new3.log_prob(train_y)
                grads = tape.gradient(loss_step3, get_kernel_params(gp_new3))
                optimizer_gp_val3.apply_gradients(zip(grads,get_kernel_params(gp_new3)))

            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val3,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val3,
                    predictive_noise_variance=0.)
            
            pred = gprm.mean()
            rmse3 += (pred[-1]-test_y[0])**2 

        else:
            my_cluster4.append(Z)
            train_X, train_y = X[0:t, :], y[0:t]
            test_X, test_y = X[t:, :], y[t:]
                
            gp_new4 = tfd.GaussianProcess(kernel=ker_val4,
                            index_points=train_X,
                            observation_noise_variance=noise_val4)
            
            optimizer_gp_val4 = tf.optimizers.Adam(learning_rate=lr_val)
            for _ in range(n_adapt):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(get_kernel_params(gp_new4))                
                    loss_step4 = -gp_new4.log_prob(train_y)
                grads = tape.gradient(loss_step4, get_kernel_params(gp_new4))
                optimizer_gp_val4.apply_gradients(zip(grads,get_kernel_params(gp_new4)))

            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val4,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val4,
                    predictive_noise_variance=0.)
            
            pred = gprm.mean()
            rmse3 += (pred[-1]-test_y[0])**2  

        count += 1

    rmse3 /= (count)
    rmse3 = tf.math.sqrt(rmse3)

    return rmse3,my_cluster1,my_cluster2,my_cluster3,my_cluster4

if __name__ == '__main__':

    last_set3 = []

    amplitude_set1 = []
    length_set1 = []
    noise_set1 = []

    amplitude_set2 = []
    length_set2 = []
    noise_set2 = []

    amplitude_set3 = []
    length_set3 = []
    noise_set3 = []

    amplitude_set4 = []
    length_set4 = []
    noise_set4 = []

    train_tensor = HR_dataloader("train")
    val_tensor = HR_dataloader("val")
    test_tensor = HR_dataloader("test")

    #Define individual parameters
    amplitude1 = tf.Variable(1.,dtype="float64",name='amplitude1', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
    length_scale1 = tf.Variable(1.,dtype="float64",name='length_scale1', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
    observation_noise_variance1 = tf.Variable(1.,dtype="float64",name='observation_noise_variance1', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

    amplitude2 = tf.Variable(1.,dtype="float64",name='amplitude2', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
    length_scale2 = tf.Variable(1.,dtype="float64",name='length_scale2', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
    observation_noise_variance2 = tf.Variable(1.,dtype="float64",name='observation_noise_variance2', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

    amplitude3 = tf.Variable(1.,dtype="float64",name='amplitude3', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
    length_scale3 = tf.Variable(1.,dtype="float64",name='length_scale3', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
    observation_noise_variance3 = tf.Variable(1.,dtype="float64",name='observation_noise_variance3', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

    amplitude4 = tf.Variable(1.,dtype="float64",name='amplitude4', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
    length_scale4 = tf.Variable(1.,dtype="float64",name='length_scale4', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
    observation_noise_variance4 = tf.Variable(1.,dtype="float64",name='observation_noise_variance4', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

    #Define global parameters 
    model_embed1 = GRU_embed(16)
    model1 = GRU(32,1)
    mean1 = Warping_mean(model1, model_embed1)

    model_embed2 = GRU_embed(16)
    model2 = GRU(32,1)
    mean2 = Warping_mean(model2, model_embed2)

    model_embed3 = GRU_embed(16)
    model3 = GRU(32,1)
    mean3 = Warping_mean(model3, model_embed3)

    model_embed4 = GRU_embed(16)
    model4 = GRU(32,1)
    mean4 = Warping_mean(model4, model_embed4)

    #Define kernel
    kernel1 = RBF_embed(amplitude=amplitude1,length_scale=length_scale1,embed_fn=model_embed1)
    kernel2 = RBF_embed(amplitude=amplitude2,length_scale=length_scale2,embed_fn=model_embed2)
    kernel3 = RBF_embed(amplitude=amplitude3,length_scale=length_scale3,embed_fn=model_embed3)
    kernel4 = RBF_embed(amplitude=amplitude4,length_scale=length_scale4,embed_fn=model_embed4)

    #Define hyperparameters
    n_epoch = 50
    n_adapt = 15
    lr = 1e-3
    optimizer_mean = tf.optimizers.Adam(learning_rate=lr)
    optimizer_ker = tf.optimizers.Adam(learning_rate=lr)
    lr_val = 1e-1

    ## Initiate
    #Suppose we have five cluster
    K = 4
    sample_size = len(train_tensor)

    # Instantiate the random pi_c
    cluster_pi = tf.ones(shape=[K,],dtype=tf.float64)/K # We expect to have K clusters 

    best_ker1,best_mean1,best_noise1,best_ker2,best_mean2,best_noise2,best_ker3,best_mean3,best_noise3,best_ker4,best_mean4,best_noise4,best_cluster1,best_cluster2,best_cluster3,best_cluster4 = train_and_valid(train_tensor,
                                                                                                                                                                                                                val_tensor,
                                                                                                                                                                                                                n_epoch,
                                                                                                                                                                                                                n_adapt,
                                                                                                                                                                                                                optimizer_mean,
                                                                                                                                                                                                                optimizer_ker,
                                                                                                                                                                                                                cluster_pi,
                                                                                                                                                                                                                kernel1,
                                                                                                                                                                                                                mean1,
                                                                                                                                                                                                                observation_noise_variance1,
                                                                                                                                                                                                                kernel2,
                                                                                                                                                                                                                mean2,
                                                                                                                                                                                                                observation_noise_variance2,
                                                                                                                                                                                                                kernel3,
                                                                                                                                                                                                                mean3,
                                                                                                                                                                                                                observation_noise_variance3,
                                                                                                                                                                                                                kernel4,
                                                                                                                                                                                                                mean4,
                                                                                                                                                                                                                observation_noise_variance4,
                                                                                                                                                                                                                lr_val,
                                                                                                                                                                                                                K)

    eval_ker1 = copy.deepcopy(best_ker1)
    eval_mean1 = copy.deepcopy(best_mean1)
    eval_noise1 = copy.deepcopy(best_noise1)
    eval_ker2 = copy.deepcopy(best_ker2)
    eval_mean2 = copy.deepcopy(best_mean2)
    eval_noise2 = copy.deepcopy(best_noise2)
    eval_ker3 = copy.deepcopy(best_ker3)
    eval_mean3 = copy.deepcopy(best_mean3)
    eval_noise3 = copy.deepcopy(best_noise3)
    eval_ker4 = copy.deepcopy(best_ker4)
    eval_mean4 = copy.deepcopy(best_mean4)
    eval_noise4 = copy.deepcopy(best_noise4)

    last_loss3,test_cluster1,test_cluster2,test_cluster3,test_cluster4=evaluate(test_tensor,15,lr_val,eval_ker1,eval_mean1,eval_noise1,eval_ker2,eval_mean2,eval_noise2,eval_ker3,eval_mean3,eval_noise3,eval_ker4,eval_mean4,eval_noise4)
    
    last_set3.append(copy.deepcopy(last_loss3))

    amplitude_set1.append(tf.constant(copy.deepcopy(eval_ker1.amplitude)))
    amplitude_set2.append(tf.constant(copy.deepcopy(eval_ker2.amplitude)))
    amplitude_set3.append(tf.constant(copy.deepcopy(eval_ker3.amplitude)))
    amplitude_set4.append(tf.constant(copy.deepcopy(eval_ker4.amplitude)))

    length_set1.append(tf.constant(copy.deepcopy(eval_ker1.length_scale)))
    length_set2.append(tf.constant(copy.deepcopy(eval_ker2.length_scale)))
    length_set3.append(tf.constant(copy.deepcopy(eval_ker3.length_scale)))
    length_set4.append(tf.constant(copy.deepcopy(eval_ker4.length_scale)))

    noise_set1.append(tf.constant(copy.deepcopy(eval_noise1)))
    noise_set2.append(tf.constant(copy.deepcopy(eval_noise2)))
    noise_set3.append(tf.constant(copy.deepcopy(eval_noise3)))
    noise_set4.append(tf.constant(copy.deepcopy(eval_noise4)))

    print(f"Mean of Loss one = {np.mean(last_set3)}")

    print(f"Mean of amplitude1 = {np.mean(amplitude_set1)}")
    print(f"Mean of amplitude2 = {np.mean(amplitude_set2)}")
    print(f"Mean of amplitude3 = {np.mean(amplitude_set3)}")
    print(f"Mean of amplitude4 = {np.mean(amplitude_set4)}")

    print(f"Mean of length1 = {np.mean(length_set1)}")
    print(f"Mean of length2 = {np.mean(length_set2)}")
    print(f"Mean of length3 = {np.mean(length_set3)}")
    print(f"Mean of length4 = {np.mean(length_set4)}")

    print(f"Mean of noise1 = {np.mean(noise_set1)}")
    print(f"Mean of noise2 = {np.mean(noise_set2)}")
    print(f"Mean of noise3 = {np.mean(noise_set3)}")
    print(f"Mean of noise4 = {np.mean(noise_set4)}")

    with open('./HR/p4_indi_Loss.txt', 'wb') as f:
        pickle.dump((last_set3,amplitude_set1,amplitude_set2,amplitude_set3,amplitude_set4,length_set1,length_set2,length_set3,length_set4,noise_set1,noise_set2,noise_set3,noise_set4), f)

