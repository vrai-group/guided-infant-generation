import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import preprocess_input
import math
sys.path.insert(1, '../utils')
from utils import utils_wgan
from skimage.transform import resize
from scipy.linalg import sqrtm

Config_file = __import__('1_config_utils')
config = Config_file.Config()


####### LOSS
# Fuinzione di loss input: y_true, y_pred
def PoseMaskLoss1(output_G1, image_raw_1, image_raw_0, mask_1, mask_0):
    image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)
    image_raw_0 = tf.cast(image_raw_0, dtype=tf.float32)
    mask_1 = tf.cast(mask_1, dtype=tf.float32)
    mask_0 = tf.cast(mask_0, dtype=tf.float32)
    mask_0_inv = 1 - mask_0

    # La PoseMakLoss1  è quella implementata sul paper
    primo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1))  # L1 loss
    #primo_membro = 0.005 * tf.reduce_mean(tf.abs(output_G1 - image_raw_0) * mask_0_inv)  # L1 loss
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

#piu il valore è basso più le geenrate sono uguali alle reali
def fid_score(output_G1, image_raw_1, mean_0, mean_1):

    def scale_images(images, new_shape):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)

    def compute_embeddings(images):
        inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                            weights="imagenet",
                                                            pooling='avg',
                                                            input_shape=(299,299,3))
        embeddings = inception_model.predict(images)

        return embeddings

    def calculate_fid(real_embeddings, generated_embeddings):
        # calculate mean and covariance statistics
        mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False).reshape(1,-1)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False).reshape(1,-1)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = np.sqrt(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid

    image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
    output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
    output_G1 = tf.cast(output_G1, dtype=tf.float16)

    image_raw_1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0,
                                           clip_value_max=32765), dtype=tf.float32)
    output_G1 = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(output_G1, mean_0, 32765.5), clip_value_min=0,
                                         clip_value_max=32765), dtype=tf.float32)

    image_raw_1_3channel = tf.concat([image_raw_1, image_raw_1, image_raw_1], axis=-1)
    output_G1_3channel = tf.concat([output_G1, output_G1, output_G1], axis=-1)
    image_raw_1_3channel_p = preprocess_input(image_raw_1_3channel)
    output_G1_3channel_p = preprocess_input(output_G1_3channel)
    image_raw_1_3channel_p = scale_images(image_raw_1_3channel_p, (299, 299, 3))
    output_G1_3channel_p = scale_images(output_G1_3channel_p, (299, 299, 3))

    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(image_raw_1_3channel_p)

    # compute embeddings for generated images
    generated_image_embeddings = compute_embeddings(output_G1_3channel_p)

    fid = calculate_fid(real_image_embeddings, generated_image_embeddings)

    return fid

#### Learning rate
def step_decay(epoch):
    initial_lrate = config.lr_initial_G1
    drop_rate = config.drop_rate_G1
    epoch_rate = config.lr_update_epoch_G1 #ogni quanto eseguire l aggiornamento
    return initial_lrate * (drop_rate ** math.floor(epoch/epoch_rate))

# Optimizer
def optimizer():
    return tf.keras.optimizers.Adam(learning_rate=config.lr_initial_G1, beta_1=0.5, beta_2=0.999)

###### MODEL
def build_model():
    #####Encoder
    encoder_layer_list = []
    inputs = keras.Input(shape=config.input_shape_G1)
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

    return model







