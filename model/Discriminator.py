import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import RandomUniform

####### LOSS
def Loss(D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0):

    # Fake
    fake = 0.25 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_neg_refined_result, labels=tf.zeros_like(D_neg_refined_result))) \
           + 0.25 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_neg_image_raw_0, labels=tf.zeros_like(D_neg_image_raw_0)))

    # Real
    real = 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_pos_image_raw_1, labels=tf.ones_like(D_pos_image_raw_1)))

    loss = fake + real

    return loss, fake, real


# Optimizer
def optimizer():
    return tf.keras.optimizers.Adam(learning_rate=config.lr_initial_D, beta_1=0.5)

###### MODEL
def build_model(config):

    inputs = Input(shape=config.input_shape_d)

    # Primo layer
    x = Conv2D(filters=64, kernel_size=5, strides = (2,2),
                kernel_initializer=RandomUniform(minval=-0.02 * np.sqrt(3), maxval=0.02 * np.sqrt(3)),
                bias_initializer='zeros', padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    # Secondo layer
    x = Conv2D(filters=64 * 2, kernel_size=5, strides =(2, 2),
               kernel_initializer=RandomUniform(minval=-0.02 * np.sqrt(3), maxval=0.02 * np.sqrt(3)),
               bias_initializer='zeros',padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Terzo layer
    x = Conv2D(filters=64 * 4, kernel_size=5, strides =(2, 2),
               kernel_initializer=RandomUniform(minval=-0.02 * np.sqrt(3), maxval=0.02 * np.sqrt(3)),
               bias_initializer='zeros',padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Quarto layer
    x = Conv2D(filters=64 * 8, kernel_size=5, strides =(2, 2),
               kernel_initializer=RandomUniform(minval=-0.02 * np.sqrt(3), maxval=0.02 * np.sqrt(3)),
               bias_initializer='zeros',padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Fully
    x = Reshape([-1, 6 * 8 * 8*64])(x)  # [batch*3, 6*8*512]
    outputs = Dense(units=1, activation= None, kernel_initializer="glorot_uniform", bias_initializer="zeros")(x) # [batch*4, 1]

    model = keras.Model(inputs, outputs)

    return model







