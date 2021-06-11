import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import RandomUniform

####### LOSS
def Loss(D_z_pos, D_z_neg, D_neg_image_raw_0):

    #Fake
    fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z_neg, labels=tf.zeros_like(D_z_neg)))

    #Real
    real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z_pos, labels=tf.ones_like(D_z_pos)))

    loss = fake + real

    return loss / 2, fake, real

###### METRICA
def mse(Y, output_G1):
    return None

# Optimizer
def optimizer():
    return tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5)

###### MODEL
def build_model(config):

    #TODO manca inseriemnto regolarizzaione weights conv e dense
    #TODO manca da controllare se la batch normalization corrisponde con quello usato nel paper

    inputs = Input(shape=(128,64,3))

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
    x = Reshape([-1, 8 * 4 * 8*64])(x)  # [batch*3, 8*4*512]
    outputs = Dense(units=1, activation= None, kernel_initializer="glorot_uniform", bias_initializer="zeros")(x) # [batch*4, 1]

    model = keras.Model(inputs, outputs)

    return model







