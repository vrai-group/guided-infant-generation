import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

def inception_preprocess_image(image, mean):
    def scale_images(images, new_shape):
        new_image = resize(images[0], new_shape, 0)
        new_img = resize(new_image, new_shape, 0)
        v = np.empty((1, new_shape[0], new_shape[1], new_shape[2]))
        v[0] = new_img
        return v

    image = tf.reshape(image, [-1, 96, 128, 1])
    image = tf.cast(
        tf.cast(tf.clip_by_value(unprocess_image(image, mean, 32765.5), clip_value_min=0,
                                 clip_value_max=32765), dtype=tf.uint8), dtype=tf.float32)

    image_3channel = tf.concat([image, image, image], axis=-1)
    image_3channel = scale_images(image_3channel, (299, 299, 3))
    image_3channel_p = preprocess_input(image_3channel)

    return image_3channel_p



