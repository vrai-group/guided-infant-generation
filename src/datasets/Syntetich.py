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

    'pz_condition': tf.io.FixedLenFeature([], tf.string),  # nome del pz condition
    'pz_target': tf.io.FixedLenFeature([], tf.string),  # nome del pz target

    'Ic_image_name': tf.io.FixedLenFeature([], tf.string),  # nome img condition
    'It_image_name': tf.io.FixedLenFeature([], tf.string),  # nome img target
    'Ic': tf.io.FixedLenFeature([], tf.string),  # Immagine di condizione Ic
    'It': tf.io.FixedLenFeature([], tf.string),  # Immagine target It

    'image_height': tf.io.FixedLenFeature([], tf.int64, default_value=96),
    'image_width': tf.io.FixedLenFeature([], tf.int64, default_value=128),

    # valori delle coordinate originali della posa ridimensionati a 96x128
    'Ic_original_keypoints': tf.io.FixedLenFeature((), dtype=tf.string),
    'It_original_keypoints': tf.io.FixedLenFeature((), dtype=tf.string),
    'shape_len_Ic_original_keypoints': tf.io.FixedLenFeature([], tf.int64),
    'shape_len_It_original_keypoints': tf.io.FixedLenFeature([], tf.int64),

    # maschera binaria a radius (r_k) con shape [96, 128, 1]
    'Mc': tf.io.FixedLenFeature([96 * 128 * 1], tf.int64),
    'Mt': tf.io.FixedLenFeature([96 * 128 * 1], tf.int64),

     # Sparse tensor per la posa. Gli indici e i valori considerano il riempimento (ingrandimento) del Keypoints di raggio r_k
    'Ic_indices': tf.io.FixedLenFeature((), dtype=tf.string),
    'Ic_values': tf.io.FixedLenFeature((), dtype=tf.string),
    'It_indices': tf.io.FixedLenFeature((), dtype=tf.string),
    'It_values': tf.io.FixedLenFeature((), dtype=tf.string),
    'shape_len_Ic_indices': tf.io.FixedLenFeature([], tf.int64),
    'shape_len_It_indices': tf.io.FixedLenFeature([], tf.int64),

    'radius_keypoints': tf.io.FixedLenFeature([], tf.int64),
}

# ritorna un TF.data
def get_unprocess_dataset(name_tfrecord):
    def _decode_function(example_proto): # funzione di decodifica
        example = tf.io.parse_single_example(example_proto, example_description)

        # NAME
        name_condition = example['Ic_image_name']
        name_target = example['It_image_name']

        # PZ
        pz_condition = example['pz_condition']
        pz_target = example['pz_target']

        # ORIGINAL_PEAKS
        shape_len_Ic_original_keypoints = example['shape_len_Ic_original_keypoints']
        Ic_original_keypoints = tf.reshape(tf.io.decode_raw(example['Ic_original_keypoints'], tf.int64),
                                      [shape_len_Ic_original_keypoints, 2])

        shape_len_It_original_keypoints = example['shape_len_It_original_keypoints']
        It_original_keypoints = tf.reshape(tf.io.decode_raw(example['It_original_keypoints'], tf.int64),
                                      [shape_len_It_original_keypoints, 2])

        # INDICES E VALUES
        shape_len_Ic_indices = example['shape_len_Ic_indices']
        Ic_indices = tf.reshape(tf.io.decode_raw(example['Ic_indices'], tf.int64), [shape_len_Ic_indices, 3])
        Ic_values = tf.io.decode_raw(example['Ic_values'], tf.int64)

        shape_len_It_indices = example['shape_len_It_indices']
        indices_1 = tf.reshape(tf.io.decode_raw(example['It_indices'], tf.int64), [shape_len_It_indices, 3])
        values_1 = tf.io.decode_raw(example['It_values'], tf.int64)

        # IMAGE
        Ic = tf.reshape(tf.io.decode_raw(example['Ic'], tf.uint16), [96, 128, 1])
        It = tf.reshape(tf.io.decode_raw(example['It'], tf.uint16), [96, 128, 1])

        # POSE
        Pc = tf.sparse.SparseTensor(indices=Ic_indices, values=Ic_values, dense_shape=[96, 128, 14])
        Pt = tf.sparse.SparseTensor(indices=indices_1, values=values_1, dense_shape=[96, 128, 14])

        # POSE_MASK
        Mc = tf.reshape(example['pose_mask_r4_0'], (96, 128, 1))
        Mt = tf.reshape(example['pose_mask_r4_1'], (96, 128, 1))

        # RADIUS KEY
        radius_keypoints = example['radius_keypoints']

        return Ic, It, Pc, Pt, Mc, Mt, pz_condition, pz_target, name_condition, name_target, Ic_indices, \
               indices_1, Ic_values, values_1, Ic_original_keypoints, It_original_keypoints, radius_keypoints

    reader = tf.data.TFRecordDataset(name_tfrecord)
    dataset = reader.map(_decode_function, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def preprocess_dataset(unprocess_dataset):
    def _preprocess(Ic, It, Pc, Pt, Mc, Mt, pz_condition, pz_target, name_condition, name_target,
                    indices_0, indices_1, values_0, values_1, original_peaks_0, original_peaks_1, radius_keypoints):

        mean_condition = tf.cast(tf.reduce_mean(Ic), dtype=tf.float16)
        mean_target = tf.cast(tf.reduce_mean(It), dtype=tf.float16)
        Ic = process_image(tf.cast(Ic, dtype=tf.float16), mean_condition, 32765.5)
        It = process_image(tf.cast(It, dtype=tf.float16), mean_target, 32765.5)

        Pt = tf.cast(tf.sparse.to_dense(Pt, default_value=0, validate_indices=False), dtype=tf.float16)
        Pt = Pt * 2
        Pt = tf.math.subtract(Pt, 1, name=None)  # rescale tra [-1, 1]

        Mt = tf.cast(tf.reshape(Mt, (96, 128, 1)), dtype=tf.float16)
        Mc = tf.cast(tf.reshape(Mc, (96, 128, 1)), dtype=tf.float16)

        return Ic, It, Pt, Mt, Mc, pz_condition, pz_target, name_condition, name_target, mean_condition, mean_target


    return unprocess_dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)


