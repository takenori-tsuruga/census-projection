# %%
from tf_rwkv_model import Build_RWKV_Model
import mlflow
from tqdm import tqdm
import polars as pl
import numpy as np
import tensorflow as tf
import keras as keras
from keras import layers, backend
from tf_utils import tf_init_config, DataGenerator
import pyreadr


# %%
# initialize tensorflow config
tf_init_config()
policy = keras.mixed_precision.Policy('float32')
keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# %%
# load data
df = pyreadr.read_r('data/census_training_prepped_std.rds')
df = df[None]
df = pl.from_pandas(df)
train_ds = df.filter((pl.col("calib") == 0) & (pl.col("cluster_size") > 1))
calib_ds = df.filter((pl.col("calib") == 1) & (pl.col("cluster_size") > 1))
train_ds = train_ds.drop(["calib", "cluster_size", "cluster"])
calib_ds = calib_ds.drop(["calib", "cluster_size", "cluster"])
train_ds = train_ds.filter((pl.col("year") >= 2011) & (pl.col("year") <= 2020))
calib_ds = calib_ds.filter((pl.col("year") >= 2011) & (pl.col("year") <= 2020))

raw_colnames = set(
    df.columns) - set(["year", "state", "place", "calib", "cluster_size", "cluster"])
all_colnames = set(df.columns) - set(["calib", "cluster_size", "cluster"])

# batch data generator parameters
lookback = 6
horizon = 1
step = 1
delay = 1
batch_size = 64
n_input = len(raw_colnames)
n_output = len(raw_colnames)
col_offset = 3
pred_col_range_start = col_offset
pred_col_range_end = col_offset + n_output
train_min_index = train_ds.with_row_count().filter(
    pl.col("year") == 2017).select(pl.col("row_nr")).head(1)[0, 0]
train_max_index = train_ds.with_row_count().filter(
    pl.col("year") == 2019).select(pl.col("row_nr")).tail(1)[0, 0]
calib_min_index = calib_ds.with_row_count().filter(
    pl.col("year") == 2017).select(pl.col("row_nr")).head(1)[0, 0]
calib_max_index = calib_ds.with_row_count().filter(
    pl.col("year") == 2019).select(pl.col("row_nr")).tail(1)[0, 0]

 # %%
# data generator setup
def prep_data_gen():
    train_data_gen = DataGenerator(
        data=train_ds,
        lookback=lookback,
        horizon=horizon,
        min_index=train_min_index,
        max_index=train_max_index,
        training=True,
        encoder=False,
        batch_size=batch_size,
        step=step,
        n_input=n_input,
        n_output=n_output,
        col_range_start=pred_col_range_start,
        col_range_end=pred_col_range_end
    )

    calib_data_gen = DataGenerator(
        data=calib_ds,
        lookback=lookback,
        horizon=horizon,
        min_index=calib_min_index,
        max_index=calib_max_index,
        training=False,
        encoder=False,
        batch_size=batch_size,
        step=step,
        n_input=n_input,
        n_output=n_output,
        col_range_start=pred_col_range_start,
        col_range_end=pred_col_range_end
    )

    train_data_loader = tf.data.Dataset.from_generator(
        train_data_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(batch_size, (lookback // step + 1), n_input), dtype=policy.compute_dtype),
            tf.TensorSpec(
                shape=(batch_size, n_output), dtype=policy.compute_dtype)
        )
    ).apply(tf.data.experimental.prefetch_to_device("/gpu:0", buffer_size=tf.data.AUTOTUNE))

    calib_data_loader = tf.data.Dataset.from_generator(
        calib_data_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(batch_size, (lookback // step + 1), n_input), dtype=policy.compute_dtype),
            tf.TensorSpec(
                shape=(batch_size, n_output), dtype=policy.compute_dtype)
        )
    ).apply(tf.data.experimental.prefetch_to_device("/gpu:0", buffer_size=tf.data.AUTOTUNE))

    return train_data_loader, calib_data_loader, train_data_gen, calib_data_gen


train_data_loader, calib_data_loader, train_data_gen, calib_data_gen = prep_data_gen()

# %%
# setup optimizer and tracking metrics

# optimizer = keras.optimizers.Lion()

train_loss = keras.metrics.Mean(name='train_loss')
train_s_loss = keras.metrics.Mean(name='train_s_loss')
train_q_loss = keras.metrics.Mean(name='train_q_loss')
train_m_loss = keras.metrics.Mean(name='train_m_loss')

val_loss = keras.metrics.Mean(name='val_loss')
val_s_loss = keras.metrics.Mean(name='val_s_loss')
val_q_loss = keras.metrics.Mean(name='val_q_loss')
val_m_loss = keras.metrics.Mean(name='val_m_loss')


#%%
# define loss functions
@tf.function(jit_compile=True)
def pseudo_huber_quantile(y_true, y_pred, q):
    y_pred = tf.cast(y_pred, dtype=backend.floatx())
    y_true = tf.cast(y_true, dtype=backend.floatx())
    q = tf.cast(q, dtype=backend.floatx())
    error = y_true - y_pred
    o_alpha = q
    u_alpha = 1 - q
    raw_huber_quantile = tf.where(
        error >= 0,
        tf.sqrt(tf.square(o_alpha) * tf.square(error) + 1) - 1,
        -1 * (tf.sqrt(tf.square(u_alpha) * tf.square(error) + 1) - 1)
    )
    huber_quantile = tf.abs(raw_huber_quantile)
    huber_quantile_mean = tf.abs(tf.reduce_mean(raw_huber_quantile, axis=-1, keepdims=True))
    
    return huber_quantile, huber_quantile_mean


@tf.function(jit_compile=True)
def ssim(y_true, y_pred, alpha, beta, gamma):
    y_pred = tf.cast(y_pred, dtype=backend.floatx())
    y_true = tf.cast(y_true, dtype=backend.floatx())
    mu_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    mu_true = tf.reduce_mean(y_true, axis=-1, keepdims=True)
    sigma_pred = tf.math.reduce_std(y_pred, axis=-1, keepdims=True)
    sigma_true = tf.math.reduce_std(y_true, axis=-1, keepdims=True)
    covariance = tf.reduce_mean((y_pred - mu_pred) * (y_true - mu_true), axis=-1, keepdims=True)
    luminance = (2 * mu_pred * mu_true + 1e-8) / (mu_pred ** 2 + mu_true ** 2 + 1e-8)
    contrast = (2 * sigma_pred * sigma_true + 1e-8) / (sigma_pred ** 2 + sigma_true ** 2 + 1e-8)
    structure = (covariance + 1e-8) / (sigma_pred * sigma_true + 1e-8)
    ssim = tf.pow(luminance, alpha) * tf.pow(contrast, beta) * tf.pow(structure, gamma)
    return 1 - ssim




#%%
# training step fucntions
@tf.function(jit_compile=True)
def simple_train_step(x, y_true, quantile, model, optimizer):

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        q_loss, m_loss = pseudo_huber_quantile(y_true, y_pred, quantile)
        s_loss = ssim(y_true, y_pred, 0.0, 1.0, 1.0)
        loss = q_loss + m_loss * s_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_q_loss(q_loss)
    train_s_loss(s_loss)
    train_m_loss(m_loss)

@tf.function(jit_compile=True)
def simple_val_step(x, y_true, quantile, model):
    y_pred = model(x, training=False)
    q_loss, m_loss = pseudo_huber_quantile(y_true, y_pred, quantile)
    s_loss = ssim(y_true, y_pred, 0.0, 1.0, 1.0)
    loss = q_loss + m_loss * s_loss

    val_loss(loss)
    val_q_loss(q_loss)
    val_s_loss(s_loss)
    val_m_loss(m_loss)


#%%

rwkv_50_model = Build_RWKV_Model(
    input_shape=(lookback+1, n_input),
    fcn_units=[512, 768, 1024],
    ff_activation="tanh",
    dropout_rate=0.8,
    regularlizer=None,
    output_units=n_output,
    name="rwkv_50_model"
)

# %%
# create directory for model storage
import os
if not os.path.exists('models/rwkv_50_model'):
    os.makedirs('models/rwkv_50_model')

# save model skeleton
rwkv_50_model.save('models/rwkv_50_model/skel.keras')
# compile model
rwkv_50_model.compile(jit_compile=True)
rwkv_50_optimizer = keras.optimizers.Lion()


# %%
# training loop setup
total_epochs = 1500
lr = 1e-6
# train steps per epoch
tspe = int((train_ds.shape[0] / 10 * 3) // batch_size + 1)
# validation steps per epoch
vspe = int((calib_ds.shape[0] / 10 * 3) // batch_size + 1)
min_lr = 1e-16
quantile = 0.5

# start mlflow run
mlflow.start_run(
    tags={
        "model": "rwkv_50_model"
    }
)

# training loop
epoch = 1
while epoch <= total_epochs:

    train_loss.reset_states()
    train_q_loss.reset_states()
    train_s_loss.reset_states()
    train_m_loss.reset_states()
    val_loss.reset_states()
    val_q_loss.reset_states()
    val_s_loss.reset_states()
    val_m_loss.reset_states()

    rwkv_50_optimizer.learning_rate.assign(lr)

    train_bar = tqdm(total=tspe)
    for i, gen in enumerate(train_data_loader):
        simple_train_step(gen[0], gen[1], quantile, rwkv_50_model, rwkv_50_optimizer)
        train_bar.update(1)
        train_bar.set_description(
            f'Epoch: {epoch} | lr: {lr:e} | T: {train_loss.result():e}')
        if i == tspe:
            break
    train_bar.close()

    val_bar = tqdm(total=vspe)
    for i, gen in enumerate(calib_data_loader):
        simple_val_step(gen[0], gen[1], quantile, rwkv_50_model)
        val_bar.update(1)
        val_bar.set_description(
            f'Epoch: {epoch} | lr: {lr:e} | V: {val_loss.result():e}')
        if i == vspe:
            break
    val_bar.close()

    # log metrics
    mlflow.log_metric("train_loss", train_loss.result(), step=epoch)
    mlflow.log_metric("train_q_loss", train_q_loss.result(), step=epoch)
    mlflow.log_metric("train_s_loss", train_s_loss.result(), step=epoch)
    mlflow.log_metric("train_m_loss", train_m_loss.result(), step=epoch)
    mlflow.log_metric("val_loss", val_loss.result(), step=epoch)
    mlflow.log_metric("val_q_loss", val_q_loss.result(), step=epoch)
    mlflow.log_metric("val_s_loss", val_s_loss.result(), step=epoch)
    mlflow.log_metric("val_m_loss", val_m_loss.result(), step=epoch)
    mlflow.log_metric("learning_rate", rwkv_50_optimizer.lr.numpy(), step=epoch)

    # save model checkpoints
    rwkv_50_model.save_weights('models/rwkv_50_model/epc_cpt')

    epoch += 1
    train_data_gen.reset()
    calib_data_gen.reset()
    lr = lr * 0.99


mlflow.end_run()

# remove model from memory
del rwkv_50_model


#%%
# training step fucntions
@tf.function(jit_compile=True)
def simple_train_step(x, y_true, quantile, model, optimizer):

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        q_loss, m_loss = pseudo_huber_quantile(y_true, y_pred, quantile)
        s_loss = ssim(y_true, y_pred, 0.0, 1.0, 1.0)
        loss = q_loss + m_loss * s_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_q_loss(q_loss)
    train_s_loss(s_loss)
    train_m_loss(m_loss)

#%%

rwkv_51_model = Build_RWKV_Model(
    input_shape=(lookback+1, n_input),
    fcn_units=[512, 768, 1024],
    ff_activation="tanh",
    dropout_rate=0.8,
    regularlizer=None,
    output_units=n_output,
    name="rwkv_51_model"
)

# %%
# create directory for model storage
import os
if not os.path.exists('models/rwkv_51_model'):
    os.makedirs('models/rwkv_51_model')

# save model skeleton
rwkv_51_model.save('models/rwkv_51_model/skel.keras')
# compile model
rwkv_51_model.compile(jit_compile=True)
rwkv_51_optimizer = keras.optimizers.Lion()


# %%
# training loop setup
total_epochs = 1500
lr = 1e-6
# train steps per epoch
tspe = int((train_ds.shape[0] / 10 * 3) // batch_size + 1)
# validation steps per epoch
vspe = int((calib_ds.shape[0] / 10 * 3) // batch_size + 1)
min_lr = 1e-16
quantile = 0.51

# start mlflow run
mlflow.start_run(
    tags={
        "model": "rwkv_51_model"
    }
)

# training loop
epoch = 1
while epoch <= total_epochs:

    train_loss.reset_states()
    train_q_loss.reset_states()
    train_s_loss.reset_states()
    train_m_loss.reset_states()
    val_loss.reset_states()
    val_q_loss.reset_states()
    val_s_loss.reset_states()
    val_m_loss.reset_states()

    rwkv_51_optimizer.learning_rate.assign(lr)

    train_bar = tqdm(total=tspe)
    for i, gen in enumerate(train_data_loader):
        simple_train_step(gen[0], gen[1], quantile, rwkv_51_model, rwkv_51_optimizer)
        train_bar.update(1)
        train_bar.set_description(
            f'Epoch: {epoch} | lr: {lr:e} | T: {train_loss.result():e}')
        if i == tspe:
            break
    train_bar.close()

    val_bar = tqdm(total=vspe)
    for i, gen in enumerate(calib_data_loader):
        simple_val_step(gen[0], gen[1], quantile, rwkv_51_model)
        val_bar.update(1)
        val_bar.set_description(
            f'Epoch: {epoch} | lr: {lr:e} | V: {val_loss.result():e}')
        if i == vspe:
            break
    val_bar.close()

    # log metrics
    mlflow.log_metric("train_loss", train_loss.result(), step=epoch)
    mlflow.log_metric("train_q_loss", train_q_loss.result(), step=epoch)
    mlflow.log_metric("train_s_loss", train_s_loss.result(), step=epoch)
    mlflow.log_metric("train_m_loss", train_m_loss.result(), step=epoch)
    mlflow.log_metric("val_loss", val_loss.result(), step=epoch)
    mlflow.log_metric("val_q_loss", val_q_loss.result(), step=epoch)
    mlflow.log_metric("val_s_loss", val_s_loss.result(), step=epoch)
    mlflow.log_metric("val_m_loss", val_m_loss.result(), step=epoch)
    mlflow.log_metric("learning_rate", rwkv_51_optimizer.lr.numpy(), step=epoch)

    # save model checkpoints
    rwkv_51_model.save_weights('models/rwkv_51_model/epc_cpt')

    epoch += 1
    train_data_gen.reset()
    calib_data_gen.reset()
    lr = lr * 0.99


mlflow.end_run()

del rwkv_51_model

#%%
# training step fucntions
@tf.function(jit_compile=True)
def simple_train_step(x, y_true, quantile, model, optimizer):

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        q_loss, m_loss = pseudo_huber_quantile(y_true, y_pred, quantile)
        s_loss = ssim(y_true, y_pred, 0.0, 1.0, 1.0)
        loss = q_loss + m_loss * s_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_q_loss(q_loss)
    train_s_loss(s_loss)
    train_m_loss(m_loss)

#%%

rwkv_49_model = Build_RWKV_Model(
    input_shape=(lookback+1, n_input),
    fcn_units=[512, 768, 1024],
    ff_activation="tanh",
    dropout_rate=0.8,
    regularlizer=None,
    output_units=n_output,
    name="rwkv_49_model"
)

# %%
# create directory for model storage
import os
if not os.path.exists('models/rwkv_49_model'):
    os.makedirs('models/rwkv_49_model')

# save model skeleton
rwkv_49_model.save('models/rwkv_49_model/skel.keras')
# compile model
rwkv_49_model.compile(jit_compile=True)
rwkv_49_optimizer = keras.optimizers.Lion()


# %%
# training loop setup
total_epochs = 1500
lr = 1e-6
# train steps per epoch
tspe = int((train_ds.shape[0] / 10 * 3) // batch_size + 1)
# validation steps per epoch
vspe = int((calib_ds.shape[0] / 10 * 3) // batch_size + 1)
min_lr = 1e-16
quantile = 0.49

# start mlflow run
mlflow.start_run(
    tags={
        "model": "rwkv_49_model"
    }
)

# training loop
epoch = 1
while epoch <= total_epochs:

    train_loss.reset_states()
    train_q_loss.reset_states()
    train_s_loss.reset_states()
    train_m_loss.reset_states()
    val_loss.reset_states()
    val_q_loss.reset_states()
    val_s_loss.reset_states()
    val_m_loss.reset_states()

    rwkv_49_optimizer.learning_rate.assign(lr)

    train_bar = tqdm(total=tspe)
    for i, gen in enumerate(train_data_loader):
        simple_train_step(gen[0], gen[1], quantile, rwkv_49_model, rwkv_49_optimizer)
        train_bar.update(1)
        train_bar.set_description(
            f'Epoch: {epoch} | lr: {lr:e} | T: {train_loss.result():e}')
        if i == tspe:
            break
    train_bar.close()

    val_bar = tqdm(total=vspe)
    for i, gen in enumerate(calib_data_loader):
        simple_val_step(gen[0], gen[1], quantile, rwkv_49_model)
        val_bar.update(1)
        val_bar.set_description(
            f'Epoch: {epoch} | lr: {lr:e} | V: {val_loss.result():e}')
        if i == vspe:
            break
    val_bar.close()

    # log metrics
    mlflow.log_metric("train_loss", train_loss.result(), step=epoch)
    mlflow.log_metric("train_q_loss", train_q_loss.result(), step=epoch)
    mlflow.log_metric("train_s_loss", train_s_loss.result(), step=epoch)
    mlflow.log_metric("train_m_loss", train_m_loss.result(), step=epoch)
    mlflow.log_metric("val_loss", val_loss.result(), step=epoch)
    mlflow.log_metric("val_q_loss", val_q_loss.result(), step=epoch)
    mlflow.log_metric("val_s_loss", val_s_loss.result(), step=epoch)
    mlflow.log_metric("val_m_loss", val_m_loss.result(), step=epoch)
    mlflow.log_metric("learning_rate", rwkv_49_optimizer.lr.numpy(), step=epoch)

    # save model checkpoints
    rwkv_49_model.save_weights('models/rwkv_49_model/epc_cpt')

    epoch += 1
    train_data_gen.reset()
    calib_data_gen.reset()
    lr = lr * 0.99


mlflow.end_run()

del rwkv_49_model

#%%