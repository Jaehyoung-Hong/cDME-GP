import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans

psd_kernels = tfp.math.psd_kernels
tfd = tfp.distributions

def make_data(initial,final,point_num,scale,GP_ratio=1,noise_ratio=1):

    train_X_pre = []
    train_Y_pre = []
    train_Z_pre = []

    #For checking clustering accuracy
    train_iter = 0

    #Noise~N(0,1)
    pure_noise_dist = tfp.distributions.Normal(loc=0,scale=1)

    #GP~N(0,K(1,1))
    kernel_data = psd_kernels.ExponentiatedQuadratic(
        amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
        length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'))

    #Mixing rate 0.6 for exp(X)
    for _ in range(420):
        # Unif sample X from [ini,fin]
        train_X = np.random.uniform(initial,final,point_num)
        train_X.sort()
        gp_data = tfd.GaussianProcess(kernel_data, train_X.reshape(-1,1))
        train_Y = gp_data.sample(1)
        train_X_pre.append(train_X)
        pure_noise = pure_noise_dist.sample(point_num)
        train_Y_pre.append(noise_ratio*pure_noise.numpy().ravel()+GP_ratio*train_Y.numpy().ravel()+(np.exp(train_X)).ravel()/scale)
        train_Z_pre.append(train_iter)
        train_iter += 1

    #Mixing rate 0.4 for exp(fin-X)
    for _ in range(280):
        # Unif sample X from [ini,fin]
        train_X = np.random.uniform(initial,final,point_num)
        train_X.sort()
        gp_data = tfd.GaussianProcess(kernel_data, train_X.reshape(-1,1))
        train_Y = gp_data.sample(1)
        train_X_pre.append(train_X)
        pure_noise = pure_noise_dist.sample(point_num)
        train_Y_pre.append(noise_ratio*pure_noise.numpy().ravel()+GP_ratio*train_Y.numpy().ravel()+(np.exp(final-train_X)).ravel()/scale)
        train_Z_pre.append(train_iter)
        train_iter += 1

    #Train data for models except kDME-GP    
    train_tensor = tf.data.Dataset.from_tensor_slices((train_X_pre,train_Y_pre,train_Z_pre))

    #Kmeans clustering
    train_X_pre = np.asarray(train_X_pre)
    train_Y_pre = np.asarray(train_Y_pre)
    kmeans = KMeans(n_clusters=2).fit(train_Y_pre)
    check_cluster1 = np.where(kmeans.labels_ == 0)
    check_cluster2 = np.where(kmeans.labels_ == 1)

    check_cluster1 = check_cluster1[0]
    check_cluster2 = check_cluster2[0]

    check_cluster1 = np.asarray(check_cluster1)
    check_cluster2 = np.asarray(check_cluster2)

    train_Z1 = check_cluster1
    train_Z2 = check_cluster2

    train_X1 = train_X_pre[check_cluster1]
    train_Y1 = train_Y_pre[check_cluster1]
    train_X2 = train_X_pre[check_cluster2]
    train_Y2 = train_Y_pre[check_cluster2]

    train_X1 = tf.convert_to_tensor(train_X1,dtype=tf.float64)
    train_Y1 = tf.convert_to_tensor(train_Y1,dtype=tf.float64)
    train_X2 = tf.convert_to_tensor(train_X2,dtype=tf.float64)
    train_Y2 = tf.convert_to_tensor(train_Y2,dtype=tf.float64) 

    #Train data for kDME-GP    
    train_tensor1 = tf.data.Dataset.from_tensor_slices((train_X1,train_Y1))
    train_tensor2 = tf.data.Dataset.from_tensor_slices((train_X2,train_Y2))

    val_X_pre = []
    val_Y_pre = []
    val_Z_pre = []

    val_iter = 0

    for _ in range(90):
        # Unif sample X from [ini,fin]
        train_X = np.random.uniform(initial,final,point_num)
        train_X.sort()
        gp_data = tfd.GaussianProcess(kernel_data, train_X.reshape(-1,1))
        train_Y = gp_data.sample(1)
        val_X_pre.append(train_X)
        pure_noise = pure_noise_dist.sample(point_num)
        val_Y_pre.append(noise_ratio*pure_noise.numpy().ravel()+GP_ratio*train_Y.numpy().ravel()+(np.exp(train_X)).ravel()/scale)
        val_Z_pre.append(val_iter)
        val_iter += 1

    for _ in range(60):
        # Unif sample X from [ini,fin]
        train_X = np.random.uniform(initial,final,point_num)
        train_X.sort()
        gp_data = tfd.GaussianProcess(kernel_data, train_X.reshape(-1,1))
        train_Y = gp_data.sample(1)
        val_X_pre.append(train_X)
        pure_noise = pure_noise_dist.sample(point_num)
        val_Y_pre.append(noise_ratio*pure_noise.numpy().ravel()+GP_ratio*train_Y.numpy().ravel()+(np.exp(final-train_X)).ravel()/scale)
        val_Z_pre.append(val_iter)
        val_iter += 1

    val_tensor = tf.data.Dataset.from_tensor_slices((val_X_pre,val_Y_pre,val_Z_pre))

    test_X_pre = []
    test_Y_pre = []
    test_Z_pre = []

    test_iter = 0

    for _ in range(90):
        # Unif sample X from [ini,fin]
        train_X = np.random.uniform(initial,final,point_num)
        train_X.sort()
        gp_data = tfd.GaussianProcess(kernel_data, train_X.reshape(-1,1))
        train_Y = gp_data.sample(1)
        test_X_pre.append(train_X)
        pure_noise = pure_noise_dist.sample(point_num)
        test_Y_pre.append(noise_ratio*pure_noise.numpy().ravel()+GP_ratio*train_Y.numpy().ravel()+(np.exp(train_X)).ravel()/scale)
        test_Z_pre.append(test_iter)
        test_iter += 1

    for _ in range(60):
        # Unif sample X from [ini,fin]
        train_X = np.random.uniform(initial,final,point_num)
        train_X.sort()
        gp_data = tfd.GaussianProcess(kernel_data, train_X.reshape(-1,1))
        train_Y = gp_data.sample(1)
        test_X_pre.append(train_X)
        pure_noise = pure_noise_dist.sample(point_num)
        test_Y_pre.append(noise_ratio*pure_noise.numpy().ravel()+GP_ratio*train_Y.numpy().ravel()+(np.exp(final-train_X)).ravel()/scale)
        test_Z_pre.append(test_iter)
        test_iter += 1

    test_tensor = tf.data.Dataset.from_tensor_slices((test_X_pre,test_Y_pre,test_Z_pre))
    
    return train_tensor,train_tensor1,train_tensor2,val_tensor,test_tensor,train_Z1,train_Z2