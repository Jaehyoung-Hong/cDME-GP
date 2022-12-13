import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=hidden_dim,activation='sigmoid')
        self.fc2 = tf.keras.layers.Dense(units=output_dim,activation=None)

    def call(self, inputs):
        inputs = tf.dtypes.cast(inputs,tf.float32)
        out = self.fc1(inputs)
        out = self.fc2(out)
        out = tf.squeeze(out,axis=1)
        out = tf.dtypes.cast(out,tf.float64)
        return out

class MLP_embed(tf.keras.Model):
    def __init__(self, feature_dim):
        super(MLP_embed, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=feature_dim,activation='sigmoid')

    def call(self, inputs):
        out = tf.dtypes.cast(inputs,tf.float32)
        out = self.fc1(inputs)
        out = tf.dtypes.cast(out,tf.float64)
        return out

class RNN(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=hidden_dim,return_sequences=True,return_state=True)
        self.fc = tf.keras.layers.Dense(units=output_dim,activation=None)

    def call(self, inputs):
        inputs = tf.dtypes.cast(inputs,tf.float32)
        inputs = tf.expand_dims(inputs,0)
        out,_,_ = self.lstm(inputs) 
        out = self.fc(out[0,:,:])
        out = tf.squeeze(out,axis=1)
        out = tf.dtypes.cast(out,tf.float64)
        return out

class RNN_embed(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(RNN_embed, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=hidden_dim,return_sequences=True,return_state=True)

    def call(self, inputs):
        inputs = tf.dtypes.cast(inputs,tf.float32)
        inputs = tf.expand_dims(inputs,0)
        out,_,_ = self.lstm(inputs) 
        out = tf.dtypes.cast(out,tf.float64)
        return out[0,:,:]

class GRU_embed(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(GRU_embed, self).__init__()
        self.gru = tf.keras.layers.GRU(units=hidden_dim,return_sequences=True,return_state=True)

    def call(self, inputs):
        inputs = tf.dtypes.cast(inputs,tf.float32)
        inputs = tf.expand_dims(inputs,0)
        out,_, = self.gru(inputs) 
        out = tf.dtypes.cast(out,tf.float64)
        return out[0,:,:]

class GRU(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(GRU, self).__init__()
        self.gru = tf.keras.layers.GRU(units=hidden_dim,return_sequences=True,return_state=True)
        self.fc = tf.keras.layers.Dense(units=output_dim,activation=None)

    def call(self, inputs):
        inputs = tf.dtypes.cast(inputs,tf.float32)
        inputs = tf.expand_dims(inputs,0)
        out,_ = self.gru(inputs)
        out = self.fc(out[0,:,:])
        out = tf.squeeze(out,axis=1)
        out = tf.dtypes.cast(out,tf.float64)
        return out

class Warping_mean(tf.keras.Model):
    def __init__(self, mean_fn, iwarping_fn):
        super(Warping_mean, self).__init__()
        self.mean_fn = mean_fn
        self.iwarping_fn = iwarping_fn

    def call(self, inputs):
        out = self.iwarping_fn(inputs)
        out = self.mean_fn(out)
        # out = tf.transpose(out,perm=[1,0])
        return out