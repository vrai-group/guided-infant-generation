import sys

import tensorflow as tf

import numpy as np
import pickle
import pdb
import glob
import cv2


# questa funzione controlla quanti pair abbiamo per ogni id di persona nel pickle file
def debug_pairs_pickle_file():
    name_pairs_file = '../data/p_pairs_train.p'
    with open(name_pairs_file, 'rb') as f:
        pairs = pickle.load(f, encoding='bytes')
        # pairs = pairs[:int(len(pairs)/20)]
        print("Log: lunghezza pairs: ", len(pairs))
        print(pairs[20000:25600])

    dict = {}
    for cnt, p in enumerate(pairs):
        id_i = p[0][0:4]
        if id_i in dict:
            dict[id_i] = dict[id_i] + 1
        else:
            dict[id_i] = 1

    for id in dict:
        print(id, ':', dict[id])


# questa funzione controlla quanti pair abbiamo per ogni id di persona nel dataset file
def debug_pairs_dataset_file():
    raw_dataset = tf.data.TFRecordDataset('../')

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

    # raw_dataset = tf.data.TFRecordDataset('../data/tfrecord/radius_10/BabyPose_train.tfrecord')
    raw_dataset = tf.data.TFRecordDataset('../data/tfrecord/new/New_BabyPose_train.tfrecord')
    #raw_dataset = tf.data.TFRecordDataset(' C:/Users/canna/Desktop/New_BabyPose_valid.tfrecord')

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    image_feature_description = {
        'pz_0': tf.io.FixedLenFeature([], tf.string),  # nome del pz
        'pz_1': tf.io.FixedLenFeature([], tf.string),

        'image_name_0': tf.io.FixedLenFeature([], tf.string),  # nome img
        'image_name_1': tf.io.FixedLenFeature([], tf.string),
        'image_raw_0': tf.io.FixedLenFeature([], tf.string),  # condizioni al contorno
        'image_raw_1': tf.io.FixedLenFeature([], tf.string),  # GT

        # 'image_format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
        'image_height': tf.io.FixedLenFeature([], tf.int64, default_value=96),
        'image_width': tf.io.FixedLenFeature([], tf.int64, default_value=128),

        'pose_peaks_0': tf.io.FixedLenFeature([8 * 16 * 14], tf.float32),
        'pose_peaks_1': tf.io.FixedLenFeature([8 * 16 * 14], tf.float32),

        'pose_mask_r4_0': tf.io.FixedLenFeature([96 * 128 * 1], tf.int64),
        'pose_mask_r4_1': tf.io.FixedLenFeature([96 * 128 * 1], tf.int64),

        'indices_r4_0': tf.io.FixedLenFeature((), dtype=tf.string),
        'values_r4_0': tf.io.FixedLenFeature((), dtype=tf.string),
        'indices_r4_1': tf.io.FixedLenFeature((), dtype=tf.string),
        'values_r4_1': tf.io.FixedLenFeature((), dtype=tf.string),
        'shape_len_indices_0': tf.io.FixedLenFeature([], tf.int64),
        'shape_len_indices_1': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_image_dataset = raw_dataset.map(_parse_image_function)
    cnt=0
    for image_features in parsed_image_dataset:
        cnt += 1
        print(cnt)

        if cnt >= 11:

            pz_0 = image_features['pz_0'].numpy().decode('utf-8')
            pz_1 = image_features['pz_1'].numpy().decode('utf-8')
            if pz_0 == "pz5" and pz_1 == "pz4":
                image_name_0 = image_features['image_name_0'].numpy().decode('utf-8')
                image_name_1 = image_features['image_name_1'].numpy().decode('utf-8')
                print(pz_0, "  ", pz_1)

                print(image_name_0, "  ", image_name_1)
                print("  ")
                image_raw_0 = tf.reshape(tf.io.decode_raw(image_features['image_raw_0'], tf.uint16), [96, 128, 1]).numpy()
                image_raw_1 = tf.reshape(tf.io.decode_raw(image_features['image_raw_1'], tf.uint16), [96, 128, 1]).numpy()
                # path= '../data/Babypose/'+str(pz_0)+'/'+image_name_0
                # sedici_bit = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # sedici_bit = cv2.resize(sedici_bit, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)
                # diff = sedici_bit - image_raw_0
                # print(np.max(diff))
                # print(np.max(image_raw_0))
                # print(np.max(sedici_bit))
                #Peaks
                shape_len_indices_0 = image_features['shape_len_indices_0']
                indices_r4_0 = tf.io.decode_raw(image_features['indices_r4_0'], np.int64)
                indices_r4_0 = tf.reshape(indices_r4_0, [shape_len_indices_0, 3])
                values_r4_0 = tf.io.decode_raw(image_features['values_r4_0'], np.int64)
                pose_0 = tf.sparse.SparseTensor(indices=indices_r4_0, values=values_r4_0, dense_shape=[96, 128, 14])
                pose_0 = tf.sparse.to_dense(pose_0, default_value=0, validate_indices=False)
                pose_0 = tf.math.reduce_sum(pose_0, axis=-1).numpy().reshape(96,128,1)
                #print(np.amax(values_r4_0.numpy())) # è una lista di 1
                #print(np.amax(pose_0.numpy())) # mi restituisce 3 perche ad esempio nella prima immagine i Keypoint sulla testa si sovrappongono
                shape_len_indices_1 = image_features['shape_len_indices_1']
                indices_r4_1 = tf.io.decode_raw(image_features['indices_r4_1'], np.int64)
                indices_r4_1 = tf.reshape(indices_r4_1, [shape_len_indices_1, 3])
                values_r4_1 = tf.io.decode_raw(image_features['values_r4_1'], np.int64)
                pose_1 = tf.sparse.SparseTensor(indices=indices_r4_1, values=values_r4_1, dense_shape=[96, 128, 14])
                pose_1 = tf.sparse.to_dense(pose_1, default_value=0, validate_indices=False)
                pose_1 = tf.reshape(pose_1[:, :, 7], (96, 128, 1)).numpy()
                #pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96,128,1)
                #mask
                pose_mask_r4_0 = image_features['pose_mask_r4_0'].numpy().reshape(96, 128, 1)
                pose_mask_r4_1 = image_features['pose_mask_r4_1'].numpy().reshape(96, 128, 1)
                fig = plt.figure(figsize=(10, 2))
                columns = 6
                rows = 1
                #imgs = [image_raw_0, image_raw_1, pose_0, pose_1, pose_mask_r4_0 * 255, pose_mask_r4_1 * 255]
                imgs = [image_raw_0 + pose_0*255, image_raw_1 +  pose_1*255, image_raw_0 + pose_mask_r4_0*255, image_raw_1 + pose_mask_r4_1*255,  pose_mask_r4_0 * 255, pose_mask_r4_1 * 255]
                for i in range(1, columns * rows + 1):
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(imgs[i - 1])
                plt.show()


    print(cnt)



if __name__ == "__main__":
    view_tfrecord()
