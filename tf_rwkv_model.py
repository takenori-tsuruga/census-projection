# Code containing custom keras layers and model building function for RWKV model

import tensorflow as tf
import keras as keras
from keras import layers, backend
from keras.src.layers.rnn import rnn_utils
from keras.src.utils import tf_utils

@keras.saving.register_keras_serializable()
class FCN(tf.keras.layers.Layer):
    def __init__(
            self, 
            dense_units, 
            activation="linear",
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            use_bias=True, 
            **kwargs):
        super(FCN, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.activation = keras.activations.get(activation)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.use_bias = use_bias

        self.dense_list = []
        for u in self.dense_units:
            self.dense_list.append(layers.Dense(
                u, activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                use_bias=self.use_bias))


    def call(self, x):
        for layer in self.dense_list:
            x = layer(x)

        return x
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "dense_units": self.dense_units,
            "activation": keras.activations.serialize(self.activation),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
            "use_bias": self.use_bias,
        }
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class RWKV_Time_Mixing_Block(tf.keras.layers.Layer):
    """
    Implementation of following code in keras serializable layer

    def time_mixing(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout):
        k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
        v = Wv @ ( x * mix_v + last_x * (1 - mix_v) )
        r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )

        wkv = (last_num + exp(bonus + k) * v) /      \
            (last_den + exp(bonus + k))
        rwkv = sigmoid(r) * wkv

        num = exp(-exp(decay)) * last_num + exp(k) * v
        den = exp(-exp(decay)) * last_den + exp(k)

        return Wout @ rwkv, (x,num,den)


    """
    def __init__(self, dim, att_dim, kernel_regularizer, activity_regularizer, name=None, **kwargs):
        super(RWKV_Time_Mixing_Block, self).__init__(name=name, **kwargs)
        self.dim = dim
        self.att_dim = att_dim
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.key_dense = tf.keras.layers.Dense(
            self.att_dim, 
            activation='linear', 
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False)
        self.value_dense = tf.keras.layers.Dense(
            self.att_dim, 
            activation='linear', 
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False)
        self.receptance_dense = tf.keras.layers.Dense(
            self.att_dim, 
            activation='sigmoid', 
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer,
            use_bias=False)
        self.output_dense = tf.keras.layers.Dense(
            self.dim, 
            activation='linear',
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False)
        self.initializer = tf.keras.initializers.Constant(.5)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.mix_k = self.add_weight(
            "mix_k",
            shape=(1, self.input_dim), 
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        self.mix_v = self.add_weight(
            "mix_v",
            shape=(1, self.input_dim), 
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        self.mix_r = self.add_weight(
            "mix_r",
            shape=(1, self.input_dim), 
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        self.decay = self.add_weight(
            "decay",
            shape=(1, self.att_dim),
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        self.bonus = self.add_weight(
            "bonus",
            shape=(1, self.att_dim),
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        
    def call(self, x, last_x, last_num, last_den):

        if x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            x = tf.cast(x, dtype=self._compute_dtype_object)
        if last_x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_x = tf.cast(last_x, dtype=self._compute_dtype_object)
        if last_num.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_num = tf.cast(last_num, dtype=self._compute_dtype_object)
        if last_den.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_den = tf.cast(last_den, dtype=self._compute_dtype_object)

        
        k = x * self.mix_k + last_x * (1 - self.mix_k)
        v = x * self.mix_v + last_x * (1 - self.mix_v)
        r = x * self.mix_r + last_x * (1 - self.mix_r)
        k = self.key_dense(k)
        v = self.value_dense(v)
        r = self.receptance_dense(r)

        wkv = (last_num + tf.exp(tf.cast(self.bonus, dtype=self._compute_dtype_object) + k) * v) / (last_den + tf.exp(tf.cast(self.bonus, dtype=self._compute_dtype_object) + k))
        rwkv = r * wkv

        num = tf.exp(-tf.exp(tf.cast(self.decay, dtype=self._compute_dtype_object))) * last_num + tf.exp(k) * v
        den = tf.exp(-tf.exp(tf.cast(self.decay, dtype=self._compute_dtype_object))) * last_den + tf.exp(k)

        hidden = self.output_dense(rwkv)

        return hidden, x, num, den
    
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'dim': self.dim,
            'att_dim': self.att_dim,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer)
        }
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class RWKV_Channel_Mixing_Block(tf.keras.layers.Layer):
    """
    Implementation of following code in keras serializable layer

    def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
        k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
        r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )
        vk = Wv @ np.maximum(k, 0)**2
        return sigmoid(r) * vk, x

    """
    def __init__(self, dim, ff_dim, activation, kernel_regularizer, activity_regularizer, name=None, **kwargs):
        super(RWKV_Channel_Mixing_Block, self).__init__(name=name, **kwargs)
        self.dim = dim
        self.ff_dim = ff_dim
        self.activation = keras.activations.get(activation)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.key_dense = tf.keras.layers.Dense(
            self.ff_dim,
            activation=self.activation,
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer,
            use_bias=False)
        self.receptance_dense = tf.keras.layers.Dense(
            self.dim, 
            activation='sigmoid',
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer,
            use_bias=False)
        self.value_dense = tf.keras.layers.Dense(
            self.dim, 
            activation='linear',
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False)
        self.initializer = tf.keras.initializers.Constant(.5)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.mix_k = self.add_weight(
            "mix_k",
            shape=(1, self.input_dim), 
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)
        self.mix_r = self.add_weight(
            "mix_r",
            shape=(1, self.input_dim), 
            initializer=self.initializer, 
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)

    def call(self, x, last_x):
        if x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            x = tf.cast(x, dtype=self._compute_dtype_object.base_dtype)
        if last_x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_x = tf.cast(last_x, dtype=self._compute_dtype_object.base_dtype)
        
        k = tf.multiply(x, self.mix_k) + tf.multiply(last_x, (1 - self.mix_k))
        r = x * self.mix_r + last_x * (1 - self.mix_r)
        k = self.key_dense(k)
        r = self.receptance_dense(r)
        vk = self.value_dense(k)
        hidden = r * vk
        return hidden, x
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'dim': self.dim,
            'ff_dim': self.ff_dim,
            'activation': keras.activations.serialize(self.activation),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer)
        }
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


    

@keras.saving.register_keras_serializable()
class RWKVCell(tf.keras.layers.Layer):

    def __init__(
        self,
        dim,
        ff_dim,
        att_dim,
        ff_activation,
        kernel_regularizer,
        bias_regularizer,
        activity_regularizer,
        dropout_rate,
        name=None,
        **kwargs
    ):
        super(RWKVCell, self).__init__(name=name, **kwargs)
        self.dim = dim
        self.ff_dim = ff_dim
        self.att_dim = att_dim
        self.ff_activation = keras.activations.get(ff_activation)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.dropout_rate = dropout_rate
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.time_mixing = RWKV_Time_Mixing_Block(
            dim=self.dim,
            att_dim=self.att_dim,
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer
        )

        self.channel_mixing = RWKV_Channel_Mixing_Block(
            dim=self.dim,
            ff_dim=self.ff_dim,
            activation=self.ff_activation,
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer
        )
        
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.layer_add = tf.keras.layers.Add()


    @property
    def state_size(self):
        return [tf.TensorShape([self.dim]), tf.TensorShape([self.att_dim]), tf.TensorShape([self.att_dim]), tf.TensorShape([self.dim])]

    @property
    def output_size(self):
        return tf.TensorShape([self.dim])
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, states):
        inputs_dtype = inputs.dtype
        x = inputs
        if x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            x = tf.cast(x, self._compute_dtype_object)
        last_x_1, last_num, last_den, last_x_2 = states
        last_x_1_dtype = last_x_1.dtype
        last_num_dtype = last_num.dtype
        last_den_dtype = last_den.dtype
        last_x_2_dtype = last_x_2.dtype
        if last_x_1.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_x_1 = tf.cast(last_x_1, self._compute_dtype_object)
        if last_num.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_num = tf.cast(last_num, self._compute_dtype_object)
        if last_den.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_den = tf.cast(last_den, self._compute_dtype_object)
        if last_x_2.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            last_x_2 = tf.cast(last_x_2, self._compute_dtype_object)
        x_ = self.layer_norm_1(x)
        dx, last_x_1, last_num, last_den = \
            self.time_mixing(x_, last_x_1, last_num, last_den)
        dx = self.dropout_1(dx)
        x = self.layer_add([inputs, dx])

        x_ = self.layer_norm_2(x)
        dx, last_x_2 = self.channel_mixing(x_, last_x_2)
        dx = self.dropout_2(dx)
        x = self.layer_add([x, dx])

        x = tf.cast(x, dtype=inputs_dtype)
        last_x_1 = tf.cast(last_x_1, dtype=last_x_1_dtype)
        last_num = tf.cast(last_num, dtype=last_num_dtype)
        last_den = tf.cast(last_den, dtype=last_den_dtype)
        last_x_2 = tf.cast(last_x_2, dtype=last_x_2_dtype)

        states = [last_x_1, last_num, last_den, last_x_2]

        return x, states
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return rnn_utils.generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype
        )
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'dim': self.dim,
            'ff_dim': self.ff_dim,
            'att_dim': self.att_dim,
            'ff_activation': keras.activations.serialize(self.ff_activation),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
            'dropout_rate': self.dropout_rate
        }
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)




def Build_RWKV_Model(
        input_shape,
        fcn_units,
        ff_activation,
        dropout_rate,
        regularlizer,
        output_units,
        name="RWKV_Model"
):
    inputs = keras.Input(shape=input_shape)
    x = FCN(dense_units=fcn_units,
            activation="linear",
            kernel_regularizer=regularlizer,
            bias_regularizer=regularlizer,
            )(inputs)
    x = layers.RNN(
        RWKVCell(
            dim=fcn_units[-1],
            ff_dim=fcn_units[-1]*4,
            att_dim=fcn_units[-1],
            ff_activation=ff_activation,
            kernel_regularizer=regularlizer,
            bias_regularizer=regularlizer,
            activity_regularizer=regularlizer,
            dropout_rate=dropout_rate
        ),
        return_sequences=False,
    )(x)
    x = layers.Dense(
        units=output_units, 
        activation="linear",
        kernel_regularizer=regularlizer,
        bias_regularizer=regularlizer,
        name="Output"
        )(x)
    model = keras.Model(inputs, x, name=name)
    return model

