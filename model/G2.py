import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy


# Fuinzione di loss
def Loss(D_z_neg, refined_result, image_raw_1, mask_1):

    # Loss per imbrogliare il discriminatore creando un immagine sempre più reale
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z_neg, labels=tf.ones_like(D_z_neg)))

    primo_membro = tf.reduce_mean(tf.abs(refined_result - image_raw_1))  # L1 loss
    secondo_membro = tf.reduce_mean(tf.abs(refined_result - image_raw_1) * mask_1)
    PoseMaskLoss2 = primo_membro + secondo_membro

    loss = gen_cost + PoseMaskLoss2*10

    return loss


# Metrica
def mse(Y, output_G1):
    None

# Optimizer
def optimizer():
    return tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5)


def build_model(config):
    #####Encoder
    encoder_layer_list = []
    inputs = keras.Input(shape=config.input_shape_g2)
    x = Conv2D(config.conv_hidden_num, 3, (1, 1), padding='same', activation=config.activation_fn, data_format=config.data_format)(inputs)

    for idx in range(config.repeat_num-2):
        channel_num = config.conv_hidden_num * (idx + 1)
        x = Conv2D(channel_num, 3, (1, 1), padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        x = Conv2D(channel_num, 3, (1, 1), padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        if idx > 0:
            encoder_layer_list.append(x)
        if idx < config.repeat_num -2 - 1:
            # ho cambiato da 3 a 2 la grandeza del Kenrel se non non portano le misura con l originale
            x = Conv2D(channel_num, 2, (2, 2), activation=config.activation_fn, data_format=config.data_format)(x)

    ##### Decoder
    for idx in range(config.repeat_num-2):
        if idx < config.repeat_num -2 - 1:
            x = Concatenate(axis=-1)([x, encoder_layer_list[-1-idx]])  # Long Skip connestion
        x = Conv2D(config.conv_hidden_num, 3, 1, padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        x = Conv2D(config.conv_hidden_num, 3, 1, padding='same', activation=config.activation_fn, data_format=config.data_format)(x)
        if idx < config.repeat_num -2 - 1:
            # x = slim.layers.conv2d_transpose(x, hidden_num * (config.repeat_num-idx-1), 3, 2, activation_fn=activation_fn, data_format=data_format)
            # Effettua l upscale delle feturemaps
            x = UpSampling2D(size=(2, 2), data_format=config.data_format, interpolation="nearest")(x)


    outputs = Conv2D(config.input_image_raw_channel, 3, 1, padding='same', activation=None, data_format=config.data_format)(x)

    model = keras.Model(inputs, outputs)

    return model



