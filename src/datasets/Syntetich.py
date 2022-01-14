"""
Questo contiene:

- PROCESSAMENTO IMMAGINE: funzioni per processare la singola immagine
- PROCESSAMENTO TFRECORD: funzioni per preprocessare il TFrecord file
"""
import tensorflow as tf

##########################
# PROCESSAMENTO IMMAGINE
##########################
def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm

def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel


##########################
# PROCESSAMENTO TFRECORD
##########################
example_description = {

    'pz_0': tf.io.FixedLenFeature([], tf.string),  # nome del pz
    'pz_1': tf.io.FixedLenFeature([], tf.string),

    'image_name_0': tf.io.FixedLenFeature([], tf.string),  # nome img
    'image_name_1': tf.io.FixedLenFeature([], tf.string),
    'image_raw_0': tf.io.FixedLenFeature([], tf.string),  # Condizione
    'image_raw_1': tf.io.FixedLenFeature([], tf.string),  # Target

    'image_height': tf.io.FixedLenFeature([], tf.int64, default_value=96),
    'image_width': tf.io.FixedLenFeature([], tf.int64, default_value=128),

    "original_peaks_0": tf.io.FixedLenFeature((), dtype=tf.string),
    "original_peaks_1": tf.io.FixedLenFeature((), dtype=tf.string),
    'shape_len_original_peaks_0': tf.io.FixedLenFeature([], tf.int64),
    'shape_len_original_peaks_1': tf.io.FixedLenFeature([], tf.int64),

    'pose_mask_r4_0': tf.io.FixedLenFeature([96 * 128 * 1], tf.int64),
    'pose_mask_r4_1': tf.io.FixedLenFeature([96 * 128 * 1], tf.int64),

    'indices_r4_0': tf.io.FixedLenFeature((), dtype=tf.string),
    'values_r4_0': tf.io.FixedLenFeature((), dtype=tf.string),
    'indices_r4_1': tf.io.FixedLenFeature((), dtype=tf.string),
    'values_r4_1': tf.io.FixedLenFeature((), dtype=tf.string),
    'shape_len_indices_0': tf.io.FixedLenFeature([], tf.int64),
    'shape_len_indices_1': tf.io.FixedLenFeature([], tf.int64),

    'radius_keypoints': tf.io.FixedLenFeature([], tf.int64),
}

# ritorna un TF.data
def get_unprocess_dataset(name_tfrecord):
    def _decode_function(example_proto): # funzione di decodifica
        example = tf.io.parse_single_example(example_proto, example_description)

        # NAME
        name_0 = example['image_name_0']
        name_1 = example['image_name_1']

        # PZ
        pz_0 = example['pz_0']
        pz_1 = example['pz_1']

        # ORIGINAL_PEAKS
        shape_len_original_peaks_0 = example['shape_len_original_peaks_0']
        original_peaks_0 = tf.reshape(tf.io.decode_raw(example['original_peaks_0'], tf.int64),
                                      [shape_len_original_peaks_0, 2])

        shape_len_original_peaks_1 = example['shape_len_original_peaks_1']
        original_peaks_1 = tf.reshape(tf.io.decode_raw(example['original_peaks_1'], tf.int64),
                                      [shape_len_original_peaks_1, 2])

        # INDICES E VALUES
        shape_len_indices_0 = example['shape_len_indices_0']
        indices_0 = tf.reshape(tf.io.decode_raw(example['indices_r4_0'], tf.int64), [shape_len_indices_0, 3])
        values_0 = tf.io.decode_raw(example['values_r4_0'], tf.int64)

        shape_len_indices_1 = example['shape_len_indices_1']
        indices_1 = tf.reshape(tf.io.decode_raw(example['indices_r4_1'], tf.int64), [shape_len_indices_1, 3])
        values_1 = tf.io.decode_raw(example['values_r4_1'], tf.int64)

        # IMAGE
        image_raw_0 = tf.reshape(tf.io.decode_raw(example['image_raw_0'], tf.uint16), [96, 128, 1])
        image_raw_1 = tf.reshape(tf.io.decode_raw(example['image_raw_1'], tf.uint16), [96, 128, 1])

        # POSE
        pose_0 = tf.sparse.SparseTensor(indices=indices_0, values=values_0, dense_shape=[96, 128, 14])
        pose_1 = tf.sparse.SparseTensor(indices=indices_1, values=values_1, dense_shape=[96, 128, 14])

        # POSE_MASK
        mask_0 = tf.reshape(example['pose_mask_r4_0'], (96, 128, 1))
        mask_1 = tf.reshape(example['pose_mask_r4_1'], (96, 128, 1))

        # RADIUS KEY
        radius_keypoints = example['radius_keypoints']

        return image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1, pz_0, pz_1, name_0, name_1, indices_0, \
               indices_1, values_0, values_1, original_peaks_0, original_peaks_1, radius_keypoints

    reader = tf.data.TFRecordDataset(name_tfrecord)
    dataset = reader.map(_decode_function, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def preprocess_dataset(unprocess_dataset):
    def _preprocess(image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1, pz_0, pz_1, name_0, name_1,
                       indices_0, indices_1, values_0, values_1, original_peaks_0, original_peaks_1, radius_keypoints):

        mean_0 = tf.cast(tf.reduce_mean(image_raw_0), dtype=tf.float16)
        mean_1 = tf.cast(tf.reduce_mean(image_raw_1), dtype=tf.float16)
        image_raw_0 = process_image(tf.cast(image_raw_0, dtype=tf.float16), mean_0, 32765.5)
        image_raw_1 = process_image(tf.cast(image_raw_1, dtype=tf.float16), mean_1, 32765.5)

        pose_1 = tf.cast(tf.sparse.to_dense(pose_1, default_value=0, validate_indices=False), dtype=tf.float16)
        pose_1 = pose_1 * 2
        pose_1 = tf.math.subtract(pose_1, 1, name=None)  # rescale tra [-1, 1]

        mask_1 = tf.cast(tf.reshape(mask_1, (96, 128, 1)), dtype=tf.float16)
        mask_0 = tf.cast(tf.reshape(mask_0, (96, 128, 1)), dtype=tf.float16)

        return image_raw_0, image_raw_1, pose_1, mask_1, mask_0, pz_0, pz_1, name_0, name_1, mean_0, mean_1


    return unprocess_dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)


