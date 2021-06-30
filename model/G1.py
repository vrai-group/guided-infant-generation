import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.insert(1, '../utils')
from utils import utils_wgan

####### LOSS
# Fuinzione di loss input: y_true, y_pred
def PoseMaskLoss1(Y, output_G1):
    image_raw_1 = tf.reshape(Y[:, :, :, 0], [-1, 96, 128, 1])
    mask_1 = tf.reshape(Y[:, :, :, 1], [-1, 96, 128, 1])
    # print("g1",output_G1.shape)
    # print("mask1", mask_1.shape)
    # print("raw1", image_raw_1.shape)

    # La PoseMakLoss1  è quella implementata sul paper
    primo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1))  # L1 loss
    secondo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1) * mask_1)
    PoseMaskLoss1 = primo_membro + secondo_membro

    # tf.print("", "\n", output_stream=sys.stdout)
    # tf.print("L1_Loss1:", primo_membro, output_stream=sys.stdout)
    # tf.print("PoseMakLoss1:", PoseMaskLoss1, output_stream=sys.stdout)
    # tf.print("Media: ", tf.math.reduce_mean(image_raw_1[0]), output_stream=sys.stdout)

    return PoseMaskLoss1

###### METRICA
# Metrica MSE
def mse(Y, output_G1):
    image_raw_1 = tf.reshape(Y[:, :, :, 0], [-1, 96, 128, 1])
    return tf.reduce_mean(tf.square(output_G1 - image_raw_1))

# Metrica SSIM
def m_ssim(Y, output_G1):
    image_raw_1 = Y[:, :, :, 0]
    image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 1, 32765.5)
    output_G1 = utils_wgan.unprocess_image(output_G1, 1, 32765.5)

    image_raw_1 = tf.clip_by_value(image_raw_1, clip_value_min=0, clip_value_max=32765)
    image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
    output_G1 = tf.clip_by_value(output_G1, clip_value_min=0, clip_value_max=32765)

    result = tf.image.ssim(output_G1, image_raw_1, max_val=32765)
    mean = tf.reduce_mean(result)

    return mean

#### Learning rate
def step_decay(epoch):
    initial_lrate = 2e-5

    # Aggiorniamo il lr ogni epoca
    if epoch > 0:
        lrate = initial_lrate / (0.5 * epoch)
    else:
        lrate = initial_lrate

    return lrate


###### MODEL
def build_model(config):
    #####Encoder
    encoder_layer_list = []
    inputs = keras.Input(shape=config.input_shape_g1)
    x = Conv2D(config.conv_hidden_num, 3, (1, 1), padding='same', activation=config.activation_fn, data_format=config.data_format)(inputs)

    for idx in range(config.repeat_num):
        channel_num = config.conv_hidden_num * (idx + 1)
        res = x
        x = Conv2D(channel_num, 3, (1, 1), padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        x = Conv2D(channel_num, 3, (1, 1), padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        x = Add()([x, res])
        encoder_layer_list.append(x)
        if idx < config.repeat_num - 1:
            # ho cambiato da 3 a 2 la grandeza del Kenrel se non non portano le misura con l originale
            # Questo layer si preoccupa di dimezzare le dimensioni W e H
            x = Conv2D(config.conv_hidden_num * (idx + 2), 2, (2, 2), activation=config.activation_fn, data_format=config.data_format)(x)

    # output [num_batch, 98.304] --> 98.304:  12x16x512
    x = Reshape([-1, int(np.prod([config.min_fea_map_H, config.min_fea_map_W, channel_num]))])(x)
    # output [num_batch, 64]
    z = x = Dense(config.z_num, activation=None)(x)

    ##### Decoder
    # output [num batch, 24576]
    x = Dense(int(np.prod([config.min_fea_map_H, config.min_fea_map_W, config.conv_hidden_num ])), activation=None)(z)
    # output [num batch, 12,16,128]
    x = Reshape([config.min_fea_map_H, config.min_fea_map_W, config.conv_hidden_num])(x)

    for idx in range(config.repeat_num):

        x = Concatenate(axis=-1)([x, encoder_layer_list[config.repeat_num - 1 - idx]])  # Long Skip connestion
        res = x
        # channel_num = hidden_num * (config.repeat_num-idx)
        channel_num = x.get_shape()[-1]
        x = Conv2D(channel_num, 3, 1, padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        x = Conv2D(channel_num, 3, 1, padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        x = Add()([x, res])
        if idx < config.repeat_num - 1:
            # x = slim.layers.conv2d_transpose(x, hidden_num * (config.repeat_num-idx-1), 3, 2, activation_fn=activation_fn, data_format=data_format)
            # Effettua l upscale delle feturemaps
            x = UpSampling2D(size=(2, 2), data_format=config.data_format, interpolation="nearest")(x)
            x = Conv2D(config.conv_hidden_num  * (config.repeat_num - idx - 1), 1, 1, activation=config.activation_fn, data_format=config.data_format)(x)

    outputs = Conv2D(config.input_image_raw_channel, 3, 1, padding='same', activation=None, data_format=config.data_format)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5, beta_2=0.999),
        loss=PoseMaskLoss1,
        metrics=[mse,m_ssim],
    )

    return model







