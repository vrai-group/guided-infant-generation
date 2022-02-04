import os
import sys
import numpy as np
import tensorflow as tf

from utils.utils_methods import format_example
from utils.augumentation.methods import aug_shift, aug_flip, random_brightness, random_contrast, aug_rotation_angle

"""
name_tfrecord: path relativo al tfrecord
cnt_dataset: lungezza totale dataset augumentato
"""
def apply_augumentation(data_tfrecord_path, unprocess_dataset_iterator, name_dataset, len_dataset):
    name_file = name_dataset + '_augumentation.tfrecord'
    name_tfrecord = os.path.join(data_tfrecord_path, name_file)
    tfrecord_writer = tf.compat.v1.python_io.TFRecordWriter(name_tfrecord)

    cnt_dataset = 0
    sys.stdout.write("\nApplico augumentazione {name}..\n".format(name=name_dataset))
    for id_example in range(1):  # len_dataset
        sys.stdout.write('\rExample: {id} / {tot}'.format(id=id_example + 1, tot=len_dataset))

        batch = next(unprocess_dataset_iterator)
        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        Mt = batch[5]  # [batch, 96,128, 1]
        pz_condition = batch[6]  # [batch, 1]
        pz_target = batch[7]  # [batch, 1]
        name_img_condition = batch[8]  # [batch, 1]
        name_img_target = batch[9]  # [batch, 1]
        Ic_indices = batch[10]  # [batch, ]
        It_indices = batch[11]  # [batch, ]
        Ic_values = batch[12]  # [batch, ]
        It_values = batch[13]  # [batch, ]
        original_peaks_0 = batch[14]
        original_peaks_1 = batch[15]
        radius_keypoints = batch[16]

        dic_data = {

            'pz_condition': pz_condition.numpy()[0].decode('utf-8'),
            'pz_target': pz_target.numpy()[0].decode('utf-8'),

            'Ic_image_name': name_img_condition.numpy()[0].decode('utf-8'),
            'It_image_name': name_img_target.numpy()[0].decode('utf-8'),
            'Ic': Ic.numpy()[0],
            'It': It.numpy()[0],

            'image_format': 'PNG'.encode('utf-8'),
            'image_height': 96,
            'image_width': 128,

            'Ic_original_keypoints': original_peaks_0.numpy()[0],
            'It_original_keypoints': original_peaks_1.numpy()[0],

            # maschera binaria a radius (r_k) con shape [96, 128, 1]
            'Mc': Mc.numpy()[0].astype(np.uint8),
            'Mt': Mt.numpy()[0].astype(np.uint8),

            'Ic_indices': Ic_indices.numpy()[0],
            'Ic_values': Ic_values.numpy()[0],
            'It_indices': It_indices.numpy()[0],
            'It_values': It_values.numpy()[0],

            'radius_keypoints': radius_keypoints.numpy()[0]

        }
        example = format_example(dic_data)
        tfrecord_writer.write(example.SerializeToString())
        cnt_dataset += 1

        ##############################
        # Dinamic affine trasformation
        ##############################

        vec_dic_affine = []  # conterra i dic con le trasfromazioni affini

        ### Aug image_raw_1

        # Rotazione Random
        random_angles_1 = tf.random.uniform(shape=[4], minval=-91, maxval=91, dtype=tf.int64).numpy()
        for angle in random_angles_1:
            dic_data_rotate = aug_rotation_angle(dic_data.copy(), angle, indx_img="t")  # rotazione image target
            vec_dic_affine.append(dic_data_rotate)

        # Shift Random
        # Shift or
        random_shift_or_1 = tf.random.uniform(shape=[2], minval=-31, maxval=31, dtype=tf.int64).numpy()
        for shift in random_shift_or_1:
            dic_data_shifted = aug_shift(dic_data.copy(), indx_img="t", type="or", tx=shift)
            vec_dic_affine.append(dic_data_shifted)
        # Shift ver
        random_shift_ver_1 = tf.random.uniform(shape=[2], minval=-11, maxval=11, dtype=tf.int64).numpy()
        for shift in random_shift_ver_1:
            dic_data_shifted = aug_shift(dic_data.copy(), indx_img="t", type="ver", ty=shift)
            vec_dic_affine.append(dic_data_shifted)

        ### Aug image_raw_0
        # Piccole trasformaizoni
        list = vec_dic_affine.copy()
        for i, dic in enumerate(list):  # escludo l'immagine originale
            trasformation = tf.random.uniform(shape=[1], minval=0, maxval=4, dtype=tf.int64).numpy()
            if trasformation == 0:  # Nessuna trasformazione
                continue
            if trasformation == 1:  # Rotation
                angle = tf.random.uniform(shape=[1], minval=-46, maxval=46, dtype=tf.int64).numpy()[0]
                dic_new = aug_rotation_angle(dic.copy(), angle, indx_img="c")  # rotazione image raw_1
                vec_dic_affine[i] = dic_new
            if trasformation == 2:  # Shift Or
                shift = tf.random.uniform(shape=[1], minval=-21, maxval=21, dtype=tf.int64).numpy()[0]
                dic_new = aug_shift(dic.copy(), indx_img="c", type="or", tx=shift)  # rotazione image raw_1
                vec_dic_affine[i] = dic_new
            if trasformation == 3:  # Shift Ver
                shift = tf.random.uniform(shape=[1], minval=-11, maxval=11, dtype=tf.int64).numpy()[0]
                dic_new = aug_shift(dic.copy(), indx_img="c", type="ver", ty=shift)  # rotazione image raw_1
                vec_dic_affine[i] = dic_new

        ### Salvo Affine trasformation
        for dic in vec_dic_affine:
            example = format_example(dic)
            tfrecord_writer.write(example.SerializeToString())
            cnt_dataset += 1

        ###############################
        # Structural trasformation
        ###############################
        vec_dic_structural = []  # conterra i dic con le trasfromazioni random B e random C
        vec_dic_new_affine = []  # conterra i dic con le trasfromazioni random B e random C

        ############# Image raw 1 (New Affine)
        # Rotazione Random
        random_angles_1 = tf.random.uniform(shape=[4], minval=-91, maxval=91, dtype=tf.int64).numpy()
        for angle in random_angles_1:
            dic_data_rotate = aug_rotation_angle(dic_data.copy(), angle, indx_img="t")  # rotazione image raw_1
            vec_dic_new_affine.append(dic_data_rotate)

        # Shift Random
        # Shift or
        random_shift_or_1 = tf.random.uniform(shape=[2], minval=-31, maxval=31, dtype=tf.int64).numpy()
        for shift in random_shift_or_1:
            dic_data_shifted = aug_shift(dic_data.copy(), indx_img="t", type="or", tx=shift)
            vec_dic_new_affine.append(dic_data_shifted)
        # Shift ver
        random_shift_ver_1 = tf.random.uniform(shape=[2], minval=-11, maxval=11, dtype=tf.int64).numpy()
        for shift in random_shift_ver_1:
            dic_data_shifted = aug_shift(dic_data.copy(), indx_img="t", type="ver", ty=shift)
            vec_dic_new_affine.append(dic_data_shifted)

        ############# Image raw 0 (New Affine)
        # Piccole trasformaizoni
        list = vec_dic_new_affine.copy()
        for i, dic in enumerate(list):
            trasformation = tf.random.uniform(shape=[1], minval=0, maxval=4, dtype=tf.int64).numpy()
            if trasformation == 0:  # Nessuna trasformazione
                continue
            if trasformation == 1:  # Rotation
                angle = tf.random.uniform(shape=[1], minval=-46, maxval=46, dtype=tf.int64).numpy()[0]
                dic_new = aug_rotation_angle(dic.copy(), angle, indx_img="c")  # rotazione image raw_1
                vec_dic_new_affine[i] = dic_new
            if trasformation == 2:  # Shift Or
                shift = tf.random.uniform(shape=[1], minval=-21, maxval=21, dtype=tf.int64).numpy()[0]
                dic_new = aug_shift(dic.copy(), indx_img="c", type="or", tx=shift)  # rotazione image raw_1
                vec_dic_new_affine[i] = dic_new
            if trasformation == 3:  # Shift Ver
                shift = tf.random.uniform(shape=[1], minval=-11, maxval=11, dtype=tf.int64).numpy()[0]
                dic_new = aug_shift(dic.copy(), indx_img="c", type="ver", ty=shift)  # rotazione image raw_1
                vec_dic_new_affine[i] = dic_new

        ############# Image raw 1 (structural)
        # Random B
        for dic in vec_dic_new_affine:
            dic_aug = random_brightness(dic.copy(), indx_img="t")
            vec_dic_structural.append(dic_aug)

        # Random C
        for dic in vec_dic_new_affine:
            dic_aug = random_contrast(dic.copy(), indx_img="t")
            vec_dic_structural.append(dic_aug)

        ###### Image raw 0 (structural)

        for i, dic in enumerate(vec_dic_structural):
            trasformation = tf.random.uniform(shape=[1], minval=0, maxval=3, dtype=tf.int64).numpy()
            if trasformation == 0:  # Nessuna trasformazione
                continue
            if trasformation == 1:  # brightness
                dic_new = random_brightness(dic.copy(), indx_img="c")
                vec_dic_structural[i] = dic_new
            if trasformation == 2:  # Contrast
                dic_new = random_contrast(dic.copy(), indx_img="c")
                vec_dic_structural[i] = dic_new

        ### Salvo Structural trasformation
        for dic in vec_dic_structural:
            example = format_example(dic)
            tfrecord_writer.write(example.SerializeToString())
            cnt_dataset += 1

        ###############################
        # Flipping trasformation
        ###############################
        # Flip
        vec_tot_trasformation = vec_dic_affine + vec_dic_structural + [dic_data]
        for dic in vec_tot_trasformation:
            dic_aug = aug_flip(dic.copy())
            example = format_example(dic_aug)
            tfrecord_writer.write(example.SerializeToString())
            cnt_dataset += 1

    return name_tfrecord, cnt_dataset
