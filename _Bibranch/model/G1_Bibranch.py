import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

Config_file = __import__('B1_config_utils')
config = Config_file.Config()

def PoseMaskLoss1(output_G1, image_raw_1, image_raw_0, mask_1, mask_0):
    image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)
    image_raw_0 = tf.cast(image_raw_0, dtype=tf.float32)
    mask_1 = tf.cast(mask_1, dtype=tf.float32)
    mask_0 = tf.cast(mask_0, dtype=tf.float32)
    mask_0_inv = 1 - mask_0

    primo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1))  # L1 loss
    secondo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1) * mask_1)
    PoseMaskLoss1 = primo_membro + secondo_membro

    return PoseMaskLoss1

###### METRICA
# Metrica SSIM
def m_ssim(output_G1, image_raw_1, mean_0, mean_1):
    image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
    output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
    output_G1 = tf.cast(output_G1, dtype=tf.float16)

    image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
    output_G1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(output_G1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)

    result = tf.image.ssim(output_G1, image_raw_1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
    mean = tf.reduce_mean(result)

    return mean

def mask_ssim(output_G1, image_raw_1, mask_1, mean_0, mean_1):
    image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
    mask_1 = tf.reshape(mask_1, [-1, 96, 128, 1])
    output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
    output_G1 = tf.cast(output_G1, dtype=tf.float16)
    #mean_0 = tf.cast(mean_0, dtype=tf.float32)

    image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
    output_G1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(output_G1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
    mask_1 = tf.cast(mask_1, dtype=tf.uint16)

    mask_image_raw_1 = mask_1 * image_raw_1
    mask_output_G1 = mask_1 * output_G1

    result = tf.image.ssim(mask_image_raw_1, mask_output_G1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
    mean = tf.reduce_mean(result)
    mean = tf.cast(mean, dtype=tf.float32)

    return mean

#### Learning rate
def step_decay(epoch):
    initial_lrate = config.lr_initial_G1_Bibranch
    drop_rate = config.drop_rate_G1_Bibranch
    epoch_rate = config.lr_update_epoch_G1_Bibranch #ogni quanto eseguire l aggiornamento
    return initial_lrate * (drop_rate ** math.floor(epoch/epoch_rate))

# Optimizer
def optimizer():
    return tf.keras.optimizers.Adam(learning_rate=config.lr_initial_G1_Bibranch, beta_1=0.5, beta_2=0.999)

###### MODEL
def build_model():
    inputs = Input(shape = config.input_shape_G1_Bibranch)
    ##Encoder

    # Blocco 1
    Enc_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(inputs)
    # Enc_1 = BatchNormalization()(Enc_1)
    Enc_1 = Activation('relu')(Enc_1)

    branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
    # branch_1_Enc_1 = BatchNormalization()(branch_1_Enc_1)
    branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)

    branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_1_Enc_1)
    # branch_1_Enc_1 = BatchNormalization()(branch_1_Enc_1)
    branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)

    branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
    # branch_2_Enc_1 = BatchNormalization()(branch_2_Enc_1)
    branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)

    branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_2_Enc_1)
    # branch_2_Enc_1 = BatchNormalization()(branch_2_Enc_1)
    branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)

    concat_1 = concatenate([branch_1_Enc_1, branch_2_Enc_1])

    # Blocco 2
    Enc_2 = Conv2D(filters=256, kernel_size=2, strides=2)(concat_1)
    Enc_2 = BatchNormalization()(Enc_2)
    Enc_2 = Activation('relu')(Enc_2)

    branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
    # branch_1_Enc_2 = BatchNormalization()(branch_1_Enc_2)
    branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)

    branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_1_Enc_2)
    # branch_1_Enc_2 = BatchNormalization()(branch_1_Enc_2)
    branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)

    branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
    # branch_2_Enc_2 = BatchNormalization()(branch_2_Enc_2)
    branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)

    branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_2_Enc_2)
    # branch_2_Enc_2 = BatchNormalization()(branch_2_Enc_2)
    branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)

    concat_2 = concatenate([branch_1_Enc_2, branch_2_Enc_2])

    # Blocco 3
    Enc_3 = Conv2D(filters=384, kernel_size=2, strides=2)(concat_2)
    # Enc_3 = BatchNormalization()(Enc_3)
    Enc_3 = Activation('relu')(Enc_3)

    branch_1_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(Enc_3)
    # branch_1_Enc_3 = BatchNormalization()(branch_1_Enc_3)
    branch_1_Enc_3 = Activation('relu')(branch_1_Enc_3)

    branch_1_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(branch_1_Enc_3)
    # branch_1_Enc_3 = BatchNormalization()(branch_1_Enc_3)
    branch_1_Enc_3 = Activation('relu')(branch_1_Enc_3)

    branch_2_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(Enc_3)
    # branch_2_Enc_3 = BatchNormalization()(branch_2_Enc_3)
    branch_2_Enc_3 = Activation('relu')(branch_2_Enc_3)

    branch_2_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(branch_2_Enc_3)
    # branch_2_Enc_3 = BatchNormalization()(branch_2_Enc_3)
    branch_2_Enc_3 = Activation('relu')(branch_2_Enc_3)

    concat_3 = concatenate([branch_1_Enc_3, branch_2_Enc_3])

    # Blocco 4
    Enc_4 = Conv2D(filters=512, kernel_size=2, strides=2)(concat_3)
    # Enc_4 = BatchNormalization()(Enc_4)
    Enc_4 = Activation('relu')(Enc_4)

    branch_1_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(Enc_4)
    # branch_1_Enc_4 = BatchNormalization()(branch_1_Enc_4)
    branch_1_Enc_4 = Activation('relu')(branch_1_Enc_4)

    branch_1_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(branch_1_Enc_4)
    # branch_1_Enc_4 = BatchNormalization()(branch_1_Enc_4)
    branch_1_Enc_4 = Activation('relu')(branch_1_Enc_4)

    branch_2_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(Enc_4)
    # branch_2_Enc_4 = BatchNormalization()(branch_2_Enc_4)
    branch_2_Enc_4 = Activation('relu')(branch_2_Enc_4)

    branch_2_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(branch_2_Enc_4)
    # branch_2_Enc_4 = BatchNormalization()(branch_2_Enc_4)
    branch_2_Enc_4 = Activation('relu')(branch_2_Enc_4)

    concat_4 = concatenate([branch_1_Enc_4, branch_2_Enc_4])

    ## Bridge
    # output [num_batch, 98.304] --> 98.304:  12x16x512
    x = Reshape([-1, int(np.prod([12, 16, 512]))])(concat_4)
    # output [num_batch, 64]
    z = Dense(64, activation=None)(x)
    # output [num batch, 24576]
    z = Dense(int(np.prod([12, 16, 128])), activation=None)(z)
    # output [num batch, 12,16,128]
    x = Reshape([12, 16, 128])(z)

    ##### Decoder
    # Blocco 1
    long_connection_1 = Concatenate(axis=-1)([x, concat_4])

    branch_1_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(long_connection_1)
    # branch_1_Dec_1 = BatchNormalization()(branch_1_Dec_1)
    branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)

    branch_1_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_1)
    # branch_1_Dec_1 = BatchNormalization()(branch_1_Dec_1)
    branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)

    branch_2_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(long_connection_1)
    # branch_2_Dec_1 = BatchNormalization()(branch_2_Dec_1)
    branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)

    branch_2_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_1)
    # branch_2_Dec_1 = BatchNormalization()(branch_2_Dec_1)
    branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)

    Dec_1 = concatenate([branch_1_Dec_1, branch_2_Dec_1])
    Dec_1 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_1)
    Dec_1 = Conv2D(filters=384, kernel_size=1, strides=1, padding='same')(Dec_1)

    ###Blocco 2
    long_connection_2 = Concatenate(axis=-1)([Dec_1, concat_3])

    branch_1_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(long_connection_2)
    # branch_1_Dec_2 = BatchNormalization()(branch_1_Dec_2)
    branch_1_Dec_2 = Activation('relu')(branch_1_Dec_2)

    branch_1_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_2)
    # branch_1_Dec_2 = BatchNormalization()(branch_1_Dec_2)
    branch_1_Dec_2 = Activation('relu')(branch_1_Dec_2)

    branch_2_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(long_connection_2)
    # branch_2_Dec_2 = BatchNormalization()(branch_2_Dec_2)
    branch_2_Dec_2 = Activation('relu')(branch_2_Dec_2)

    branch_2_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_2)
    # branch_2_Dec_2 = BatchNormalization()(branch_2_Dec_2)
    branch_2_Dec_2 = Activation('relu')(branch_2_Dec_2)

    Dec_2 = concatenate([branch_1_Dec_2, branch_2_Dec_2])
    Dec_2 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_2)
    Dec_2 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(Dec_2)

    ###Blocco 3
    long_connection_3 = Concatenate(axis=-1)([Dec_2, concat_2])  # 512

    branch_1_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(long_connection_3)
    # branch_1_Dec_3 = BatchNormalization()(branch_1_Dec_3)
    branch_1_Dec_3 = Activation('relu')(branch_1_Dec_3)

    branch_1_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_3)
    # branch_1_Dec_3 = BatchNormalization()(branch_1_Dec_3)
    branch_1_Dec_3 = Activation('relu')(branch_1_Dec_3)

    branch_2_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(long_connection_3)
    # branch_2_Dec_3 = BatchNormalization()(branch_2_Dec_3)
    branch_2_Dec_3 = Activation('relu')(branch_2_Dec_3)

    branch_2_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_3)
    # branch_2_Dec_3 = BatchNormalization()(branch_2_Dec_3)
    branch_2_Dec_3 = Activation('relu')(branch_2_Dec_3)

    Dec_3 = concatenate([branch_1_Dec_3, branch_2_Dec_3])
    Dec_3 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_3)
    Dec_3 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(Dec_3)

    ###Blocco 4
    long_connection_4 = Concatenate(axis=-1)([Dec_3, concat_1])

    branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
    # branch_1_Dec_4 = BatchNormalization()(branch_1_Dec_4)
    branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)

    branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_4)
    branch_1_Dec_4 = BatchNormalization()(branch_1_Dec_4)
    branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)

    branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
    # branch_2_Dec_4 = BatchNormalization()(branch_2_Dec_4)
    branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)

    branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_4)
    # branch_2_Dec_4 = BatchNormalization()(branch_2_Dec_4)
    branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)

    Dec_4 = concatenate([branch_1_Dec_4, branch_2_Dec_4])  # 256

    outputs = Conv2D(1, 1, 1, padding='same', activation=None)(Dec_4)

    model = Model(inputs=inputs, outputs=outputs)

    return model
