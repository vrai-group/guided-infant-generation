import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy

sys.path.insert(1, '../utils')
from utils import utils_wgan

Config_file = __import__('1_config_utils')
config = Config_file.Config()



# Fuinzione di loss
def Loss(D_neg_refined_result, refined_result, image_raw_1, image_raw_0, mask_1, mask_0):
    image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)
    image_raw_0 = tf.cast(image_raw_0, dtype=tf.float32)
    mask_1 = tf.cast(mask_1, dtype=tf.float32)
    mask_0 = tf.cast(mask_0, dtype=tf.float32)

    # Loss per imbrogliare il discriminatore creando un immagine sempre più reale
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_neg_refined_result, labels=tf.ones_like(D_neg_refined_result)))

    primo_membro = tf.reduce_mean(tf.abs(refined_result - image_raw_1))  # L1 loss
    secondo_membro = tf.reduce_mean(tf.abs(refined_result - image_raw_1) * mask_1)
    PoseMaskLoss2 = primo_membro + secondo_membro

    loss = gen_cost + PoseMaskLoss2*10

    return loss


# Metrica
def m_ssim(refined_result, image_raw_1, mean_0, mean_1):
    image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
    refined_result = tf.reshape(refined_result, [-1, 96, 128, 1])

    image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
    refined_result = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(refined_result, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)

    result = tf.image.ssim(refined_result, image_raw_1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
    mean = tf.reduce_mean(result)

    return mean


def mask_ssim(refined_result, image_raw_1, mask_1, mean_0, mean_1):
    image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
    mask_1 = tf.reshape(mask_1, [-1, 96, 128, 1])
    refined_result = tf.reshape(refined_result, [-1, 96, 128, 1])

    image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
    refined_result = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(refined_result, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
    mask_1 = tf.cast(mask_1, dtype=tf.uint16)

    mask_image_raw_1 = mask_1 * image_raw_1
    mask_refined_result = mask_1 * refined_result

    result = tf.image.ssim(mask_image_raw_1, mask_refined_result, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
    mean = tf.reduce_mean(result)

    return mean

# Optimizer
def optimizer():
    return tf.keras.optimizers.Adam(learning_rate=config.lr_initial_G2, beta_1=0.5)


def build_model():

    #####Encoder
    inputs = keras.Input(shape=config.input_shape_G2)

    conv1 = Conv2D(config.conv_hidden_num, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(inputs)
    conv1 = Conv2D(config.conv_hidden_num, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv1)
    conv1 = Conv2D(config.conv_hidden_num, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv1)

    pool1 = Conv2D(config.conv_hidden_num * 2, 2, (2, 2), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv1)
    conv2 = Conv2D(config.conv_hidden_num * 2, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(pool1)
    conv2 = Conv2D(config.conv_hidden_num * 2, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv2) #256


    #Bridge
    pool3 = Conv2D(config.conv_hidden_num * 3, 2, (2, 2), activation=config.activation_fn,
                   data_format=config.data_format)(conv2)  # pool
    conv3 = Conv2D(config.conv_hidden_num * 3, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(pool3)
    conv3 = Conv2D(config.conv_hidden_num * 3, 3, (1, 1), padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv3)  # 384
    up4 = UpSampling2D(size=(2, 2), data_format=config.data_format, interpolation="nearest")(conv3)
    up4 = Conv2D(config.conv_hidden_num, 2, 1, padding="same", activation=config.activation_fn,
                 data_format=config.data_format)(up4)  # 128

    #####Decoder
    merge4 = Concatenate(axis=-1)([up4, conv2])  # Long Skip connestion 128+256 =384
    conv4 = Conv2D(384, 3, 1, padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(merge4)
    conv4 = Conv2D(384, 3, 1, padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv4)

    up5 = UpSampling2D(size=(2, 2), data_format=config.data_format, interpolation="nearest")(conv4)
    up5 = Conv2D(config.conv_hidden_num, 2, 1, padding="same", activation=config.activation_fn,
                 data_format=config.data_format)(up5)
    merge5 = Concatenate(axis=-1)([up5, conv1])  # Long Skip connestion 128+128
    conv5 = Conv2D(256, 3, 1, padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(merge5)
    conv5 = Conv2D(256, 3, 1, padding='same', activation=config.activation_fn,
                   data_format=config.data_format)(conv5)

    outputs = Conv2D(config.input_image_raw_channel, 1, 1, padding='same', activation=None,
                     data_format=config.data_format)(conv5)

    model = keras.Model(inputs, outputs)

    return model







