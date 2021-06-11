import sys

import tensorflow as tf

import numpy as np
import pickle
import pdb
import glob

# questa funzione controlla quanti pair abbiamo per ogni id di persona nel pickle file
def debug_pairs_pickle_file():
    name_pairs_file = '../data/p_pairs_train.p'
    with open(name_pairs_file, 'rb') as f:
        pairs = pickle.load(f, encoding='bytes')
        # pairs = pairs[:int(len(pairs)/20)]
        print("Log: lunghezza pairs: ", len(pairs))
        print(pairs[20000:25600])

    dict = {}
    for cnt,p in enumerate(pairs):
        id_i = p[0][0:4]
        if id_i in dict:
            dict[id_i] = dict[id_i] + 1
        else:
            dict[id_i] = 1


    for id in dict:
        print(id, ':', dict[id])

# questa funzione controlla quanti pair abbiamo per ogni id di persona nel dataset file
def debug_pairs_dataset_file():
    raw_dataset = tf.data.TFRecordDataset('..\data/Market1501_train_00000-of-00001.tfrecord')

    image_feature_description = {
        'image_name_0': tf.io.FixedLenFeature([], tf.string),  # nome dell immagine 0
        'image_name_1': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_dataset.map(_parse_image_function)
    dict = {}
    for image_features in parsed_image_dataset:
        name_0 = image_features['image_name_0'].numpy()
        id_0 = name_0[0:4]
        if id_0 in dict:
            dict[id_0] = dict[id_0] + 1
        else:
            dict[id_0] = 1

    for id in dict:
        print(id, ':', dict[id])




def view_tfrecord():
    import tensorflow as tf
    import numpy as np
    import IPython.display as display
    import matplotlib.pyplot as plt

    raw_dataset = tf.data.TFRecordDataset('..\data/Market1501_train_00000-of-00001.tfrecord')
    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    image_feature_description = {
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
          'indices_r4_0': tf.io.FixedLenFeature((),dtype=tf.string),
          'values_r4_0': tf.io.FixedLenFeature((),dtype=tf.string),
          'indices_r4_1': tf.io.FixedLenFeature((),dtype=tf.string),
          'values_r4_1': tf.io.FixedLenFeature((),dtype=tf.string),
          'shape_len_indices_0': tf.io.FixedLenFeature([], tf.int64),
          'shape_len_indices_1': tf.io.FixedLenFeature([], tf.int64),
        # 'pose_subs_0': tf.io.FixedLenFeature([20], tf.float32),
        # 'pose_subs_1': tf.io.FixedLenFeature([20], tf.float32),
    }

    parsed_image_dataset = raw_dataset.map(_parse_image_function)

    cnt=0
    for image_features in parsed_image_dataset:

        image_raw_0 = tf.reshape(tf.io.decode_jpeg(image_features['image_raw_0']), [128, 64, 3]).numpy()
        image_raw_1 = tf.reshape(tf.io.decode_jpeg(image_features['image_raw_1']), [128, 64, 3]).numpy()

        #Peaks
        shape_len_indices_0 = image_features['shape_len_indices_0']
        indices_r4_0 = tf.io.decode_raw(image_features['indices_r4_0'], tf.int64)
        indices_r4_0 = tf.reshape(indices_r4_0, [shape_len_indices_0, 3])
        values_r4_0 = tf.io.decode_raw(image_features['values_r4_0'], tf.int64)
        pose_0 = tf.sparse.SparseTensor(indices=indices_r4_0, values=values_r4_0,
                                        dense_shape=[128, 64, 18])
        pose_0 = tf.sparse.to_dense(pose_0, default_value=0, validate_indices=False)
        pose_0 = tf.math.reduce_sum(pose_0, axis=-1)
        #print(np.amax(values_r4_0.numpy())) # è una lista di 1
        #print(np.amax(pose_0.numpy())) # mi restituisce 3 perche ad esempio nella prima immagine i Keypoint sulla testa si sovrappongono

        shape_len_indices_1 = image_features['shape_len_indices_1']
        indices_r4_1 = tf.io.decode_raw(image_features['indices_r4_1'], tf.int64)
        indices_r4_1 = tf.reshape(indices_r4_1, [shape_len_indices_1, 3])
        values_r4_1 = tf.io.decode_raw(image_features['values_r4_1'], tf.int64)
        pose_1 = tf.sparse.SparseTensor(indices=indices_r4_1, values=values_r4_1,
                                        dense_shape=[128, 64, 18])
        pose_1 = tf.sparse.to_dense(pose_1, default_value=0, validate_indices=False)
        pose_1 = tf.math.reduce_sum(pose_1, axis=-1)

        #mask
        pose_mask_r4_0 = image_features['pose_mask_r4_0'].numpy().reshape(128, 64)
        pose_mask_r4_1 = image_features['pose_mask_r4_1'].numpy().reshape(128, 64)
        cnt = cnt + 1

        fig = plt.figure(figsize=(10, 10))
        columns = 6
        rows = 1
        imgs = [image_raw_0, image_raw_1, pose_0, pose_1, pose_mask_r4_0, pose_mask_r4_1]
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(imgs[i - 1])
        plt.show()


if __name__ == "__main__":
    view_tfrecord()