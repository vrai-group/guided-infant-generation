import os

import tensorflow as tf
import numpy as np
import IPython.display as display
import matplotlib.pyplot as plt
import os
import cv2

def view_tfrecord():
    name_tfrecord = 'Babypose_pz28.tfrecord'
    raw_dataset = tf.data.TFRecordDataset(name_tfrecord)
    name_pz=name_tfrecord.split('_')[1].split('.')[0]
    if not os.path.exists(name_pz):
        os.makedirs(name_pz)


    image_feature_description = {

        'pz_0': tf.io.FixedLenFeature([], tf.string),
        'image_name_0': tf.io.FixedLenFeature([], tf.string),  # nome dell immagine 0
        'image_raw_0': tf.io.FixedLenFeature([], tf.string),  # immagine in bytes  0
        'key': tf.io.FixedLenFeature([480 * 640 * 1], tf.int64),  # immagine con su stampati i keypoints
        #
        'pose_mask_r4_0': tf.io.FixedLenFeature([480 * 640 * 1], tf.int64),
        # maschera binaria a radius 4 con shape [128,64,1]

    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)


    parsed_image_dataset = raw_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:

        image_raw_0 = tf.reshape(tf.io.decode_jpeg(image_features['image_raw_0']), [480, 640, 1]).numpy()
        keypoints_img = image_features['key'].numpy().reshape(480,640, 1)
        name_img = str(image_features['image_name_0'].numpy())

        #mask
        pose_mask_r4_0 = image_features['pose_mask_r4_0'].numpy().reshape(480,640, 1)

        cv2.imwrite("./"+name_pz+"/"+name_img+".png", image_raw_0 + pose_mask_r4_0 * 255)
        cv2.imwrite("./"+name_pz+"/"+name_img+"1.png", keypoints_img)


if __name__ == "__main__":
    view_tfrecord()