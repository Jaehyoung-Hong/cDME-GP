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

from means_keras import Warping_mean, MLP_embed, MLP
from cutils import RBF_embed, get_mean_params,get_kernel_params, get_no_embed_params,get_mean_no_embed_params

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
            observation_noise_variance2):
            
    dataloader = dataloader.shuffle(1024*1024, reshuffle_each_iteration=False)
    cluster1 = []
    cluster2 = []

    #EM-step
    for i, (X, y, Z) in enumerate(dataloader):
        X = tf.reshape(X,[-1,1])
        if i == 0:
            mean1(X) #dummy #Initialize
            mean2(X)

        gp1 = tfd.GaussianProcess(kernel=kernel1,
                                index_points=X,
                                mean_fn=mean1,
                                observation_noise_variance=observation_noise_variance1)

        gp2 = tfd.GaussianProcess(kernel=kernel2,
                                index_points=X,
                                mean_fn=mean2,
                                observation_noise_variance=observation_noise_variance2)

        #Numerical trick for prob
        loss1_temp = gp1.log_prob(y)
        loss1 = tf.exp(loss1_temp-loss1_temp)
        loss2_temp = gp2.log_prob(y)
        loss2 = tf.exp(loss2_temp-loss1_temp)

        if i == 0:
            prob_r_inter1 = cluster_pi[0]*loss1
            prob_r_inter2 = cluster_pi[1]*loss2
            if (prob_r_inter1>=prob_r_inter2):
                cluster1.append(Z)
            else:
                cluster2.append(Z)
            prob_r = tf.convert_to_tensor([prob_r_inter1,prob_r_inter2])
            prob_r = tf.reshape(prob_r,[1,-1])
        else:
            prob_r_inter1 = cluster_pi[0]*loss1
            prob_r_inter2 = cluster_pi[1]*loss2
            if (prob_r_inter1>=prob_r_inter2):
                cluster1.append(Z)
            else:
                cluster2.append(Z)
            prob_r_inter = tf.convert_to_tensor([prob_r_inter1,prob_r_inter2])
            prob_r_inter = tf.reshape(prob_r_inter,[1,-1])
            prob_r = tf.concat((prob_r,prob_r_inter),axis=0)
        
    loss = 0.
    prob_r = tf.keras.utils.normalize(prob_r, order=1, axis=1)

    index = 0
    index1 = 0
    index2 = 0

    for count2 in range(prob_r.shape[0]):
        check_cluster = tf.where(prob_r[count2,:] == tf.math.reduce_max(prob_r[count2,:]))
        index += 1

        if check_cluster[0][0] == 0:
            index1 += 1   
        else:
            index2 += 1
    
    #M-step
    for i, (X, y, _) in enumerate(dataloader):
        X = tf.reshape(X,[-1,1])
                    
        gp1 = tfd.GaussianProcess(kernel=kernel1,
                                index_points=X,
                                mean_fn = mean1,
                                observation_noise_variance=observation_noise_variance1)

        gp2 = tfd.GaussianProcess(kernel=kernel2,
                                index_points=X,
                                mean_fn = mean2,
                                observation_noise_variance=observation_noise_variance2)
        
        #For one step, update global->individual parameters simultaneously 
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(get_mean_params(gp1)+get_mean_no_embed_params(gp2))                
            loss1 = gp1.log_prob(y)
            loss2 = gp2.log_prob(y)
            loss_step = -((prob_r[i,0] * (tf.math.log(cluster_pi[0])+loss1) 
                    + prob_r[i,1] * (tf.math.log(cluster_pi[1])+loss2)
                    ))
        grads = tape.gradient(loss_step, get_mean_params(gp1)+get_mean_no_embed_params(gp2))
        optimizer_train_mean.apply_gradients(zip(grads,get_mean_params(gp1)+get_mean_no_embed_params(gp2)))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(get_kernel_params(gp1)+get_kernel_params(gp2))                
            loss1 = gp1.log_prob(y)
            loss2 = gp2.log_prob(y)
            loss_step = -((prob_r[i,0] * (tf.math.log(cluster_pi[0])+loss1) 
                    + prob_r[i,1] * (tf.math.log(cluster_pi[1])+loss2)
                    ))
        grads = tape.gradient(loss_step, get_kernel_params(gp1)+get_kernel_params(gp2))
        optimizer_train_ker.apply_gradients(zip(grads,get_kernel_params(gp1)+get_kernel_params(gp2)))

        loss += loss_step
    loss /= (i+1)

    cluster_pi = tf.reduce_sum(prob_r,axis=0) / len(prob_r)
    print([index,index1,index2])  

    return loss,cluster_pi,cluster1,cluster2

#validation
def valid_step(data_loader,
            n_adapt,
            lr_val,
            ker_val1,
            mean_val1,
            noise_val1,
            ker_val2,
            mean_val2,
            noise_val2):
    loss = 0.
    for i, (X, y, _) in enumerate(data_loader):
        X = tf.reshape(X,[-1,1])
        if i == 0:
            mean_val1(X) #dummy
            mean_val2(X) #dummy

        train_X, train_y = X[:-1, :], y[:-1]
        test_X, test_y = X[-1:, :], y[-1:]
        # model prediction
        gp_new1 = tfd.GaussianProcess(kernel=ker_val1,
                                    index_points=train_X,
                                    mean_fn=mean_val1,
                                    observation_noise_variance=noise_val1)

        gp_new2 = tfd.GaussianProcess(kernel=ker_val2,
                                    index_points=train_X,
                                    mean_fn=mean_val2,
                                    observation_noise_variance=noise_val2)

        optimizer_gp_val1 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val2 = tf.optimizers.Adam(learning_rate=lr_val)

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

        if (loss_step1<=loss_step2):
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val1,
                    index_points=test_X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val1,
                    predictive_noise_variance=0.,
                    mean_fn = mean_val1)
        else:
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val2,
                    index_points=test_X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val2,
                    predictive_noise_variance=0.,
                    mean_fn = mean_val2)

        pred = gprm.mean()
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
                    lr_val,
                    K):

    best_val_err = float('inf')
    best_ker1 = None
    best_mean1 = None
    best_noise1 = None
    best_ker2 = None
    best_mean2 = None
    best_noise2 = None
    
    best_cluster1 = None
    best_cluster2 = None

    early_count = 0

    with tqdm(total=n_epoch) as t:
        for i in range(n_epoch):
            loss,cluster_pi,cluster1,cluster2=train_step(dataloader_train,
                                                        optimizer_train_mean,
                                                        optimizer_train_ker,
                                                        cluster_pi,
                                                        kernel1,
                                                        mean1,
                                                        observation_noise_variance1,
                                                        kernel2,
                                                        mean2,
                                                        observation_noise_variance2)

            ker_val1 = copy.deepcopy(kernel1)
            mean_val1 = copy.deepcopy(mean1)
            noise_val1 = copy.deepcopy(observation_noise_variance1)
            ker_val2 = copy.deepcopy(kernel2)
            mean_val2 = copy.deepcopy(mean2)
            noise_val2 = copy.deepcopy(observation_noise_variance2)

            error_val = valid_step(dataloader_val,n_adapt,lr_val,ker_val1,mean_val1,noise_val1,ker_val2,mean_val2,noise_val2)
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

                    best_cluster1 = copy.deepcopy(cluster1)
                    best_cluster2 = copy.deepcopy(cluster2)   
   
                    early_count = 0              
                    logging.info("Found new best error at {}".format(i))
                else:
                    early_count += 1

            if i<5:
                cluster_pi = tf.ones(shape=[K,],dtype=tf.float64)/K

            t.set_postfix(loss_and_val_err='{} and {}'.format(loss, error_val))
            print('\n')
            t.update()
            if (early_count>5)and(i>=5):
                break

    return best_ker1,best_mean1,best_noise1,best_ker2,best_mean2,best_noise2,best_cluster1,best_cluster2 

#Sequential Evaluation
def evaluate(data_loader,n_adapt,lr_val,ker_val1,mean_val1,noise_val1,ker_val2,mean_val2,noise_val2):
    rmse1 = 0.
    rmse2 = 0.
    rmse3 = 0.
    count = 0

    my_cluster1=[]
    my_cluster2=[]

    for (X, y, Z) in tqdm(data_loader):
        X = tf.reshape(X,[-1,1])
        if count == 0:
            mean_val1(X) #dummy
            mean_val2(X) #dummy
        
        train_X, train_y = X[:-1, :], y[:-1]
        test_X, test_y = X[-1:, :], y[-1:]

        # model prediction
        gp_new1 = tfd.GaussianProcess(kernel=ker_val1,
                                    index_points=train_X,
                                    mean_fn=mean_val1,
                                    observation_noise_variance=noise_val1)

        gp_new2 = tfd.GaussianProcess(kernel=ker_val2,
                                    index_points=train_X,
                                    mean_fn=mean_val2,
                                    observation_noise_variance=noise_val2)

        optimizer_gp_val1 = tf.optimizers.Adam(learning_rate=lr_val)
        optimizer_gp_val2 = tf.optimizers.Adam(learning_rate=lr_val)

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

        if (loss_step1<=loss_step2):
            my_cluster1.append(Z)
            for t in range(-3,0):
                train_X, train_y = X[0:t, :], y[0:t]
                if t == -1:
                    test_X, test_y = X[t:, :], y[t:]
                else:
                    test_X, test_y = X[t:t + 1, :], y[t:t + 1]

                gp_new1 = tfd.GaussianProcess(kernel=ker_val1,
                                index_points=train_X,
                                mean_fn=mean_val1,
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
                        index_points=test_X,
                        observation_index_points=train_X,
                        observations=train_y,
                        observation_noise_variance=noise_val1,
                        predictive_noise_variance=0.,
                        mean_fn = mean_val1)
                
                pred = gprm.mean()
                if t == -3:
                    rmse1 += (pred-test_y[0])**2
                elif t == -2:
                    rmse2 += (pred-test_y[0])**2
                else:
                    rmse3 += (pred-test_y[0])**2
        else:
            my_cluster2.append(Z)
            for t in range(-3,0):
                train_X, train_y = X[0:t, :], y[0:t]
                if t == -1:
                    test_X, test_y = X[t:, :], y[t:]
                else:
                    test_X, test_y = X[t:t + 1, :], y[t:t + 1]

                gp_new2 = tfd.GaussianProcess(kernel=ker_val2,
                                index_points=train_X,
                                mean_fn=mean_val2,
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
                        index_points=test_X,
                        observation_index_points=train_X,
                        observations=train_y,
                        observation_noise_variance=noise_val2,
                        predictive_noise_variance=0.,
                        mean_fn = mean_val2)
                
                pred = gprm.mean()
                if t == -3:
                    rmse1 += (pred-test_y[0])**2
                elif t == -2:
                    rmse2 += (pred-test_y[0])**2
                else:
                    rmse3 += (pred-test_y[0])**2  
        count += 1

    rmse1 /= (count)
    rmse2 /= (count)
    rmse3 /= (count)

    rmse1 = tf.math.sqrt(rmse1)
    rmse2 = tf.math.sqrt(rmse2)
    rmse3 = tf.math.sqrt(rmse3)

    return rmse1,rmse2,rmse3,my_cluster1,my_cluster2

if __name__ == '__main__':

    last_set1 = []
    last_set2 = []
    last_set3 = []

    amplitude_set1 = []
    length_set1 = []
    noise_set1 = []

    amplitude_set2 = []
    length_set2 = []
    noise_set2 = []

    for n_iter in [0,2,3,4,8,9]:

        train_tensor = tf.data.experimental.load(f'./train/train{n_iter}')
        val_tensor = tf.data.experimental.load(f'./val/val{n_iter}')
        test_tensor = tf.data.experimental.load(f'./test/test{n_iter}')

        #Define individual parameters
        amplitude1 = tf.Variable(1.,dtype="float64",name='amplitude1', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
        length_scale1 = tf.Variable(1.,dtype="float64",name='length_scale1', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
        observation_noise_variance1 = tf.Variable(1.,dtype="float64",name='observation_noise_variance1', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

        amplitude2 = tf.Variable(1.,dtype="float64",name='amplitude2', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
        length_scale2 = tf.Variable(1.,dtype="float64",name='length_scale2', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
        observation_noise_variance2 = tf.Variable(1.,dtype="float64",name='observation_noise_variance2', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

        #Define global parameters 
        model_embed = MLP_embed(16)
        model1 = MLP(32,1)
        mean1 = Warping_mean(model1, model_embed)

        model2 = MLP(32,1)
        mean2 = Warping_mean(model2, model_embed)

        #Define kernel
        kernel1 = RBF_embed(amplitude=amplitude1,length_scale=length_scale1,embed_fn=model_embed)
        kernel2 = RBF_embed(amplitude=amplitude2,length_scale=length_scale2,embed_fn=model_embed)

        #Define hyperparameters
        n_epoch = 100
        n_adapt = 15
        lr = 1e-3
        optimizer_mean = tf.optimizers.Adam(learning_rate=lr)
        optimizer_ker = tf.optimizers.Adam(learning_rate=lr)
        lr_val = 1e-2

        ## Initiate
        #Suppose we have five cluster
        K = 2
        sample_size = len(train_tensor)

        # Instantiate the random pi_c
        cluster_pi = tf.ones(shape=[K,],dtype=tf.float64)/K # We expect to have K clusters 

        best_ker1,best_mean1,best_noise1,best_ker2,best_mean2,best_noise2,best_cluster1,best_cluster2, = train_and_valid(train_tensor,
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
                                                                                                                        lr_val,
                                                                                                                        K)

        eval_ker1 = copy.deepcopy(best_ker1)
        eval_mean1 = copy.deepcopy(best_mean1)
        eval_noise1 = copy.deepcopy(best_noise1)
        eval_ker2 = copy.deepcopy(best_ker2)
        eval_mean2 = copy.deepcopy(best_mean2)
        eval_noise2 = copy.deepcopy(best_noise2)

        best_mean1.save_weights(f"./TOY{n_iter}/mean_1c_ckpt")
        best_mean2.save_weights(f"./TOY{n_iter}/mean_2c_ckpt")

        last_loss1,last_loss2,last_loss3,test_cluster1,test_cluster2=evaluate(test_tensor,250,lr_val,eval_ker1,eval_mean1,eval_noise1,eval_ker2,eval_mean2,eval_noise2)
        
        with open(f'./TOY{n_iter}/cCluster.txt', 'wb') as f:
            pickle.dump((best_cluster1,best_cluster2,test_cluster1,test_cluster2), f)

        last_set1.append(copy.deepcopy(last_loss1))
        last_set2.append(copy.deepcopy(last_loss2))
        last_set3.append(copy.deepcopy(last_loss3))

        amplitude_set1.append(tf.constant(copy.deepcopy(eval_ker1.amplitude)))
        amplitude_set2.append(tf.constant(copy.deepcopy(eval_ker2.amplitude)))

        length_set1.append(tf.constant(copy.deepcopy(eval_ker1.length_scale)))
        length_set2.append(tf.constant(copy.deepcopy(eval_ker2.length_scale)))

        noise_set1.append(tf.constant(copy.deepcopy(eval_noise1)))
        noise_set2.append(tf.constant(copy.deepcopy(eval_noise2)))
    
    print(f"Mean of Loss three = {np.mean(last_set1)}")
    print(f"Std of Loss three = {np.std(last_set1)}")

    print(f"Mean of Loss two = {np.mean(last_set2)}")
    print(f"Std of Loss two = {np.std(last_set2)}")

    print(f"Mean of Loss one = {np.mean(last_set3)}")
    print(f"Std of Loss one = {np.std(last_set3)}")

    print(f"Mean of amplitude1 = {np.mean(amplitude_set1)}")
    print(f"Std of amplitude1 = {np.std(amplitude_set1)}")
    print(f"Mean of amplitude2 = {np.mean(amplitude_set2)}")
    print(f"Std of amplitude2 = {np.std(amplitude_set2)}")

    print(f"Mean of length1 = {np.mean(length_set1)}")
    print(f"Std of length1 = {np.std(length_set1)}")
    print(f"Mean of length2 = {np.mean(length_set2)}")
    print(f"Std of length2 = {np.std(length_set2)}")

    print(f"Mean of noise1 = {np.mean(noise_set1)}")
    print(f"Std of noise1 = {np.std(noise_set1)}")    
    print(f"Mean of noise2 = {np.mean(noise_set2)}")
    print(f"Std of noise2 = {np.std(noise_set2)}")

    with open('./TOY/cLoss_Review.txt', 'wb') as f:
        pickle.dump((last_set1,last_set2,last_set3,amplitude_set1,amplitude_set2,length_set1,length_set2,noise_set1,noise_set2), f)

