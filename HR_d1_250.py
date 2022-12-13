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
from means_keras import RNN, RNN_embed, Warping_mean, MLP_embed, MLP, GRU, GRU_embed
from cutils import RBF_embed, get_mean_params,get_kernel_params,get_no_embed_params,get_mean_no_embed_params

#train
def train_step(dataloader,optimizer_train_mean,optimizer_train_ker,kernel,mean,observation_noise_variance):
    dataloader = dataloader.shuffle(1024*1024, reshuffle_each_iteration=True)
    loss = 0.
    for i, (X, y, _) in enumerate(dataloader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
        if i == 0:
            mean(X) #dummy #Initialize
        gp = tfd.GaussianProcess(kernel=kernel,
                                index_points=X,
                                mean_fn = mean,
                                observation_noise_variance=observation_noise_variance)
           
        #For one step, updata global->individual parameters
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(get_mean_params(gp))                
            loss_step = -gp.log_prob(y)
        grads = tape.gradient(loss_step, get_mean_params(gp))
        optimizer_train_mean.apply_gradients(zip(grads,get_mean_params(gp)))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(get_kernel_params(gp))                
            loss_step = -gp.log_prob(y)
        grads = tape.gradient(loss_step, get_kernel_params(gp))
        optimizer_train_ker.apply_gradients(zip(grads,get_kernel_params(gp)))

        loss += loss_step
    loss /= (i+1)
    return loss

#validation
def valid_step(data_loader,n_adapt,lr_val, ker_val, mean_val, noise_val):
    loss = 0.
    for i, (X, y, _) in enumerate(data_loader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
        if i == 0:
            mean_val(X) #dummy
        train_X, train_y = X[:-1, :], y[:-1]
        test_X, test_y = X[-1:, :], y[-1:]
        # model prediction
        gp_new = tfd.GaussianProcess(kernel=ker_val,
                                    index_points=train_X,
                                    mean_fn=mean_val,
                                    observation_noise_variance=noise_val)

        optimizer_gp_val = tf.optimizers.Adam(learning_rate=lr_val)

        for _ in range(n_adapt):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                variables = get_kernel_params(gp_new)
                tape.watch(variables)                
                loss_step = -gp_new.log_prob(train_y)
            grads = tape.gradient(loss_step, variables)
            optimizer_gp_val.apply_gradients(zip(grads,variables))
                
        gprm = tfd.GaussianProcessRegressionModel(
                kernel=ker_val,
                index_points=X,
                observation_index_points=train_X,
                observations=train_y,
                observation_noise_variance=noise_val,
                predictive_noise_variance=0.,
                mean_fn = mean_val)

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
                    kernel,
                    mean,
                    observation_noise_variance,
                    lr_val):

    best_val_err = float('inf')
    best_ker = None
    best_mean = None
    best_noise = None
    
    early_count = 0

    with tqdm(total=n_epoch) as t:
        for i in range(n_epoch):
            loss= train_step(dataloader_train,optimizer_train_mean,optimizer_train_ker,kernel,mean,observation_noise_variance)
            ker_val = copy.deepcopy(kernel)
            mean_val = copy.deepcopy(mean)
            noise_val = copy.deepcopy(observation_noise_variance)
            error_val = valid_step(dataloader_val,n_adapt,lr_val,ker_val,mean_val,noise_val)
            is_best = error_val <= best_val_err
            if is_best:
                best_val_err = error_val
                best_ker = copy.deepcopy(kernel)
                best_mean = copy.deepcopy(mean)
                best_noise = copy.deepcopy(observation_noise_variance)  
                early_count = 0              
                logging.info("Found new best error at {}".format(i))
            else:
                early_count += 1

            t.set_postfix(loss_and_val_err='{} and {}'.format(loss, error_val))
            print('\n')
            t.update()
            if early_count > 5:
                break
    return best_ker,best_mean,best_noise 

#Sequential Evaluation
def evaluate(data_loader,n_adapt,lr_val, ker_val, mean_val, noise_val):
    rmse1 = 0.
    rmse2 = 0.
    rmse3 = 0. 
    count = 0

    for (X, y, _) in tqdm(data_loader):
        X = X.to_tensor()
        y = y.to_tensor()
        X = X[0]
        y = tf.squeeze(y)
        if count == 0:
            mean_val(X) #dummy

        for t in range(-3,0):
            train_X, train_y = X[0:t, :], y[0:t]
            if t == -1:
                test_X, test_y = X[t:, :], y[t:]
            else:
                test_X, test_y = X[t:t + 1, :], y[t:t + 1]
        
            # model prediction
            gp_new = tfd.GaussianProcess(kernel=ker_val,
                                        index_points=train_X,
                                        mean_fn=mean_val,
                                        observation_noise_variance=noise_val)

            optimizer_gp_val = tf.optimizers.Adam(learning_rate=lr_val)

            for _ in range(n_adapt):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    variables = get_kernel_params(gp_new)
                    tape.watch(variables)                
                    loss_step = -gp_new.log_prob(train_y)
                grads = tape.gradient(loss_step, variables)
                optimizer_gp_val.apply_gradients(zip(grads,variables))
        
            gprm = tfd.GaussianProcessRegressionModel(
                    kernel=ker_val,
                    index_points=X,
                    observation_index_points=train_X,
                    observations=train_y,
                    observation_noise_variance=noise_val,
                    predictive_noise_variance=0.,
                    mean_fn = mean_val)

            pred = gprm.mean()
            if t == -3:
                rmse1 += (pred[-3]-test_y[0])**2
            elif t == -2:
                rmse2 += (pred[-2]-test_y[0])**2
            else:
                rmse3 += (pred[-1]-test_y[0])**2

        count += 1

    rmse1 /= (count)
    rmse2 /= (count)
    rmse3 /= (count)

    rmse1 = tf.math.sqrt(rmse1)
    rmse2 = tf.math.sqrt(rmse2)
    rmse3 = tf.math.sqrt(rmse3)

    return rmse1,rmse2,rmse3


if __name__ == '__main__':

    last_set1 = []
    last_set2 = []
    last_set3 = [] 

    amplitude_set = []
    length_set = []
    noise_set = []

    train_tensor = HR_dataloader("train")
    val_tensor = HR_dataloader("val")
    test_tensor = HR_dataloader("test")

    #Define individual parameters
    amplitude = tf.Variable(1.0,dtype="float64",name='amplitude', constraint=lambda t: tf.clip_by_value(t, 1e-6, 100))
    length_scale = tf.Variable(1.0,dtype="float64",name='length_scale', constraint=lambda t: tf.clip_by_value(t, 1e-1, 100))
    observation_noise_variance = tf.Variable(1.0,dtype="float64",name='observation_noise_variance', constraint=lambda t: tf.clip_by_value(t, 1e-2, 100))

    #Define global parameters 
    model_embed = GRU_embed(16)
    model = GRU(32,1)
    mean = Warping_mean(model, model_embed)

    #Define kernel
    kernel = RBF_embed(amplitude=amplitude,length_scale=length_scale,embed_fn=model_embed)

    #Define hyperparameters
    n_epoch = 50
    n_adapt = 50
    lr = 1e-3
    optimizer_mean = tf.optimizers.Adam(learning_rate=lr)
    optimizer_ker = tf.optimizers.Adam(learning_rate=lr)
    lr_val = 1e-2

    best_ker,best_mean,best_noise = train_and_valid(train_tensor,
                                                    val_tensor,
                                                    n_epoch,
                                                    n_adapt,
                                                    optimizer_mean,
                                                    optimizer_ker,
                                                    kernel,
                                                    mean,
                                                    observation_noise_variance,
                                                    lr_val)
    
    eval_ker = copy.deepcopy(best_ker)
    eval_mean = copy.deepcopy(best_mean)
    eval_noise = copy.deepcopy(best_noise)

    best_mean.save_weights("./HR/mean_d_250_ckpt")
    
    last_loss1, last_loss2, last_loss3 = evaluate(test_tensor,250,lr_val,eval_ker,eval_mean,eval_noise)

    last_set1.append(copy.deepcopy(last_loss1))
    last_set2.append(copy.deepcopy(last_loss2))
    last_set3.append(copy.deepcopy(last_loss3))

    amplitude_set.append(tf.constant(copy.deepcopy(eval_ker.amplitude)))
    length_set.append(tf.constant(copy.deepcopy(eval_ker.length_scale)))
    noise_set.append(tf.constant(copy.deepcopy(eval_noise)))

    print(f"Mean of Loss three = {np.mean(last_set1)}")
    print(f"Std of Loss three = {np.std(last_set1)}")

    print(f"Mean of Loss two = {np.mean(last_set2)}")
    print(f"Std of Loss two = {np.std(last_set2)}")

    print(f"Mean of Loss one = {np.mean(last_set3)}")
    print(f"Std of Loss one = {np.std(last_set3)}")

    print(f"Mean of amplitude = {np.mean(amplitude_set)}")
    print(f"Std of amplitude = {np.std(amplitude_set)}")

    print(f"Mean of length = {np.mean(length_set)}")
    print(f"Std of length = {np.std(length_set)}")

    print(f"Mean of noise = {np.mean(noise_set)}")
    print(f"Std of noise = {np.std(noise_set)}")    

    with open('./HR/dLoss_250.txt', 'wb') as f:
        pickle.dump((last_set1,last_set2,last_set3,amplitude_set,length_set,noise_set), f)