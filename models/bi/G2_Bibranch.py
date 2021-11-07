# import numpy as np
# import sys
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.losses import BinaryCrossentropy
#
# sys.path.insert(1, '../utils')
# from utils import utils_wgan
#
#
# # Fuinzione di loss
# def Loss(D_neg_refined_result, refined_result, image_raw_1, mask_1):
#     image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)
#     refined_result = tf.cast(refined_result, dtype=tf.float32)
#     mask_1 = tf.cast(mask_1, dtype=tf.float32)
#
#     # Loss per imbrogliare il discriminatore creando un immagine sempre più reale
#     gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_neg_refined_result, labels=tf.ones_like(D_neg_refined_result)))
#     gen_cost = tf.cast(gen_cost, dtype=tf.float32)
#
#     primo_membro = tf.reduce_mean(tf.abs(refined_result - image_raw_1))  # L1 loss
#     secondo_membro = tf.reduce_mean(tf.abs(refined_result - image_raw_1) * mask_1)
#     PoseMaskLoss2 = primo_membro + secondo_membro
#
#     loss = gen_cost + PoseMaskLoss2*10
#
#     return loss
#
#
# # Metrica
# def m_ssim(refined_result, image_raw_1, mean_0, mean_1):
#     image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
#     refined_result = tf.reshape(refined_result, [-1, 96, 128, 1])
#
#     image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
#     refined_result = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(refined_result, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
#
#     result = tf.image.ssim(refined_result, image_raw_1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
#     mean = tf.reduce_mean(result)
#
#     return mean
#
#
# def mask_ssim(refined_result, image_raw_1, mask_1, mean_0, mean_1):
#     image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
#     mask_1 = tf.reshape(mask_1, [-1, 96, 128, 1])
#     refined_result = tf.reshape(refined_result, [-1, 96, 128, 1])
#
#     image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
#     refined_result = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(refined_result, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
#     mask_1 = tf.cast(mask_1, dtype=tf.uint16)
#
#     mask_image_raw_1 = mask_1 * image_raw_1
#     mask_refined_result = mask_1 * refined_result
#
#     result = tf.image.ssim(mask_image_raw_1, mask_refined_result, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
#     mean = tf.reduce_mean(result)
#
#     return mean
#
# # Optimizer
# def optimizer():
#     return tf.keras.optimizers.Adam(learning_rate=config.lr_initial_G2, beta_1=0.5)
#
#
# def build_model():
#
#     #####Encoder
#     inputs = keras.Input(shape=config.input_shape_G2)
#
#     Enc_1 = Conv2D(128, 3, (1, 1), padding='same', activation=config.activation_fn,
#                    data_format=config.data_format)(inputs)
#
#     branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
#     branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)
#
#     branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_1_Enc_1)
#     branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)
#
#     branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
#     branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)
#
#     branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_2_Enc_1)
#     branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)
#
#     concat_1 = concatenate([branch_1_Enc_1, branch_2_Enc_1])   #128
#
#     Enc_2 = Conv2D(filters=256, kernel_size=2, strides=2)(concat_1)
#     Enc_2 = Activation('relu')(Enc_2)
#
#     branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
#     branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)
#
#     branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_1_Enc_2)
#     branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)
#
#     branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
#     branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)
#
#     branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_2_Enc_2)
#     branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)
#
#     concat_2 = concatenate([branch_1_Enc_2, branch_2_Enc_2])
#
#     #Bridge
#     pool3 = Conv2D(384, 2, (2, 2), activation=config.activation_fn,
#                    data_format=config.data_format)(concat_2)  # pool
#     conv3 = Conv2D(384, 3, (1, 1), padding='same', activation=config.activation_fn,
#                    data_format=config.data_format)(pool3)
#     conv3 = Conv2D(384, 3, (1, 1), padding='same', activation=config.activation_fn,
#                    data_format=config.data_format)(conv3)  # 384
#     up4 = UpSampling2D(size=(2, 2), data_format=config.data_format, interpolation="nearest")(conv3)
#     up4 = Conv2D(config.conv_hidden_num, 2, 1, padding="same", activation=config.activation_fn,
#                  data_format=config.data_format)(up4)  # 128
#
#     #####Decoder
#
#     #Blocco1
#     long_connection_1 = Concatenate(axis=-1)([up4, concat_2])  #128+256=384
#
#     branch_1_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(long_connection_1)
#     branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)
#
#     branch_1_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(
#         branch_1_Dec_1)
#     branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)
#
#     branch_2_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(long_connection_1)
#     branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)
#
#     branch_2_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(
#         branch_2_Dec_1)
#     branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)
#
#     Dec_1 = concatenate([branch_1_Dec_1, branch_2_Dec_1])
#     Dec_1 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_1)
#     Dec_1 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(Dec_1)
#
#     #Blocco2
#     long_connection_4 = Concatenate(axis=-1)([Dec_1, concat_1])  # 128+128=256
#
#     branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
#     branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)
#
#     branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
#         branch_1_Dec_4)
#     branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)
#
#     branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
#     branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)
#
#     branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
#         branch_2_Dec_4)
#     branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)
#
#     Dec_4 = concatenate([branch_1_Dec_4, branch_2_Dec_4])  # 256
#
#     outputs = Conv2D(1, 1, 1, padding='same', activation=None)(Dec_4)
#
#     model = keras.Model(inputs, outputs)
#
#     return model
#
#
#
