"""
Questo codice rilascia un reader per la lettura del TFRecord del dataset Market
"""
import os
import pdb
import pickle
import sys
import tensorflow as tf
from utils import utils_wgan

# Per maggiori info su tf.records vedi: https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
_FILE_PATTERN = '%s_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': None, 'test': None}



class Market():
    def __init__(self):
        self.keypoint_num = 18
        self.example_description = {
        'image_name_0': tf.io.FixedLenFeature([], tf.string),  # nome dell immagine 0
        'image_name_1': tf.io.FixedLenFeature([], tf.string),
        'image_raw_0': tf.io.FixedLenFeature([], tf.string),  # condizioni al contorno
        'image_raw_1': tf.io.FixedLenFeature([], tf.string),  # GT
        # 'label': tf.io.FixedLenFeature([], tf.int64),  # For FixedLenFeature, [] means scalar
        # 'id_0': tf.io.FixedLenFeature([], tf.int64),
        # 'id_1': tf.io.FixedLenFeature([], tf.int64),
        'cam_0': tf.io.FixedLenFeature([], tf.int64),
        'cam_1': tf.io.FixedLenFeature([], tf.int64),
        'image_format': tf.io.FixedLenFeature([], tf.string, default_value='jpg'),
        'image_height': tf.io.FixedLenFeature([], tf.int64, default_value=128),
        'image_width': tf.io.FixedLenFeature([], tf.int64, default_value=64),
        # 'real_data': tf.io.FixedLenFeature([], tf.int64, default_value=1),
        'pose_peaks_0': tf.io.FixedLenFeature([16 * 8 * 18], tf.float32),
        'pose_peaks_1': tf.io.FixedLenFeature([16 * 8 * 18], tf.float32),  # posa desiderata
        'pose_mask_r4_0': tf.io.FixedLenFeature([128 * 64 * 1], tf.int64),
        'pose_mask_r4_1': tf.io.FixedLenFeature([128 * 64 * 1], tf.int64),  # pose mask della desiderata

        # 'shape': tf.io.FixedLenFeature([1], tf.int64),
        'indices_r4_0': tf.io.FixedLenFeature((), dtype=tf.string),
        'values_r4_0': tf.io.FixedLenFeature((), dtype=tf.string),
        'indices_r4_1': tf.io.FixedLenFeature((), dtype=tf.string),
        'values_r4_1': tf.io.FixedLenFeature((), dtype=tf.string),
        'shape_len_indices_0': tf.io.FixedLenFeature([], tf.int64),
        'shape_len_indices_1': tf.io.FixedLenFeature([], tf.int64),
        # 'pose_subs_0': tf.io.FixedLenFeature([20], tf.float32),
        # 'pose_subs_1': tf.io.FixedLenFeature([20], tf.float32),
    }

    # ritorna un TF.data
    def get_unprocess_dataset(self, dataset_dir, name_tfrecord):
        # deve sempre ritornare uno o piu elementi
        def _decode_function(example_proto):
            example = tf.io.parse_single_example(example_proto, self.example_description)

            # IMAGE NAME
            image_name_0 = example['image_name_0']

            # IMAGE
            image_raw_0 = tf.reshape(tf.io.decode_jpeg(example['image_raw_0']), [128, 64, 3])
            image_raw_1 = tf.reshape(tf.io.decode_jpeg(example['image_raw_1']), [128, 64, 3])

            # POSE
            shape_len_indices_0 = example['shape_len_indices_0']
            pose_0 = tf.sparse.SparseTensor(
                indices=tf.reshape(tf.io.decode_raw(example['indices_r4_0'], tf.int64), [shape_len_indices_0, 3]),
                values=tf.io.decode_raw(example['values_r4_0'], tf.int64),
                dense_shape=[128, 64, self.keypoint_num])

            shape_len_indices_1 = example['shape_len_indices_1']
            pose_1 = tf.sparse.SparseTensor(
                indices=tf.reshape(tf.io.decode_raw(example['indices_r4_1'], tf.int64), [shape_len_indices_1, 3]),
                values=tf.io.decode_raw(example['values_r4_1'], tf.int64),
                dense_shape=[128, 64, self.keypoint_num])

            # POSE_MASK
            mask_0 = tf.reshape(example['pose_mask_r4_0'], (128, 64, 1))
            mask_1 = tf.reshape(example['pose_mask_r4_1'], (128, 64, 1))

            return image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1, image_name_0

        file_pattern = os.path.join(dataset_dir, name_tfrecord)  # poichè la sintassi del file pateern è _FILE_PATTERN = '%s_%s_*.tfrecord'
        reader = tf.data.TFRecordDataset(file_pattern)
        dataset = reader.map(_decode_function)

        return dataset

    # ritorna un TF.data preprocessato in G1
    def get_preprocess_G1_dataset(self, unprocess_dataset):
        def _preprocess_G1(image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1, image_name_0 ):

            image_raw_0 = utils_wgan.process_image(tf.compat.v1.to_float(image_raw_0), 127.5,
                                                   127.5)  # rescale in valori di [-1,1]
            image_raw_1 = utils_wgan.process_image(tf.compat.v1.to_float(image_raw_1), 127.5,
                                                   127.5)  # rescale in valori di [-1,1]

            pose_1 = tf.cast(tf.sparse.to_dense(pose_1, default_value=0, validate_indices=False), dtype=tf.float32)
            mask_1 = tf.cast(tf.reshape(mask_1, (128, 64, 1)), dtype=tf.float32)

            pose_1 = pose_1 * 2
            pose_1 = tf.math.subtract(pose_1, 1, name=None)  # rescale tra [-1, 1]

            X = tf.concat([image_raw_0, pose_1], axis=-1)
            Y = tf.concat([image_raw_1, mask_1], axis=-1)

            return X, Y

        return unprocess_dataset.map(_preprocess_G1, num_parallel_calls=tf.data.AUTOTUNE)

    # ritorna un TF.data preprocessato in G1 ma con gli output che possono essere utilizzati durante il training della GAN
    def get_preprocess_GAN_dataset(self, unprocess_dataset):
        def _preprocess_G1(image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1, image_name_0 ):
            image_raw_0 = utils_wgan.process_image(tf.compat.v1.to_float(image_raw_0), 127.5,
                                                   127.5)  # rescale in valori di [-1,1]
            image_raw_1 = utils_wgan.process_image(tf.compat.v1.to_float(image_raw_1), 127.5,
                                                   127.5)  # rescale in valori di [-1,1]

            pose_1 = tf.cast(tf.sparse.to_dense(pose_1, default_value=0, validate_indices=False), dtype=tf.float32)
            mask_1 = tf.cast(tf.reshape(mask_1, (128, 64, 1)), dtype=tf.float32)

            pose_1 = pose_1 * 2
            pose_1 = tf.math.subtract(pose_1, 1, name=None)  # rescale tra [-1, 1]

            return image_raw_0, image_raw_1, pose_1, mask_1, image_name_0
        return unprocess_dataset.map(_preprocess_G1, num_parallel_calls=tf.data.AUTOTUNE)


