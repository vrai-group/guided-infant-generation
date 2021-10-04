import os
import cv2
import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import dataset_utils

def _getSparseKeypoint(y, x, k, height, width, radius=4, mode='Solid'):
    indices = []
    values = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = np.sqrt(float(i ** 2 + j ** 2))
            if y + i >= 0 and y + i < height and x + j >= 0 and x + j < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([y + i, x + j, k])
                    values.append(1)
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('SparseKeypoint.png', dense * 255)

    return indices, values

def _getSparsePose(peaks, height, width, radius, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k] # coordinate peak ex: "300,200"
        x = p[0]
        y = p[1]
        if x != -1 and y != -1:  # non considero le occlusioni indicate con -1
            ind, val = _getSparseKeypoint(y, x, k, height, width, radius, mode)
            indices.extend(ind)
            values.extend(val)
    return indices, values

def _format_example(dic):


    example = tf.train.Example(features=tf.train.Features(feature={

        'pz_0': dataset_utils.bytes_feature(dic["pz_0"].encode('utf-8')),  # nome del pz
        'pz_1': dataset_utils.bytes_feature(dic["pz_1"].encode('utf-8')),

        'image_name_0': dataset_utils.bytes_feature(dic["image_name_0"].encode('utf-8')),  # nome dell immagine 0
        'image_name_1': dataset_utils.bytes_feature(dic["image_name_1"].encode('utf-8')),  # nome dell immagine 1
        'image_raw_0': dataset_utils.bytes_feature(dic["image_raw_0"].tostring()),  # immagine 0 in bytes
        'image_raw_1': dataset_utils.bytes_feature(dic["image_raw_1"].tostring()),  # immagine 1 in bytes

        'image_format': dataset_utils.bytes_feature('PNG'.encode('utf-8')),
        'image_height': dataset_utils.int64_feature(96),
        'image_width': dataset_utils.int64_feature(128),

        'original_peaks_0': dataset_utils.bytes_feature(np.array(dic["original_peaks_0"]).astype(np.int64).tostring()),
        'original_peaks_1': dataset_utils.bytes_feature(np.array(dic["original_peaks_1"]).astype(np.int64).tostring()),
        'shape_len_original_peaks_0': dataset_utils.int64_feature(np.array(dic["original_peaks_0"]).shape[0]),
        'shape_len_original_peaks_1': dataset_utils.int64_feature(np.array(dic["original_peaks_1"]).shape[0]),

        'pose_mask_r4_0': dataset_utils.int64_feature(dic["pose_mask_r4_0"].astype(np.uint16).flatten().tolist()),
        # maschera binaria a radius 4 con shape [96, 128, 1]
        'pose_mask_r4_1': dataset_utils.int64_feature(dic["pose_mask_r4_1"].astype(np.uint16).flatten().tolist()),
        # maschera binaria a radius 4 con shape [96, 128, 1]

        'indices_r4_0': dataset_utils.bytes_feature(np.array(dic["indices_r4_0"]).astype(np.int64).tostring()),
        # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'values_r4_0': dataset_utils.bytes_feature(np.array(dic["values_r4_0"]).astype(np.int64).tostring()),
        # coordinate a radius 4 dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'indices_r4_1': dataset_utils.bytes_feature(np.array(dic["indices_r4_1"]).astype(np.int64).tostring()),
        # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 1, servono per ricostruire il vettore di sparse [num_indices, 3]
        'values_r4_1': dataset_utils.bytes_feature(np.array(dic["values_r4_1"]).astype(np.int64).tostring()),
        'shape_len_indices_0': dataset_utils.int64_feature(np.array(dic["indices_r4_0"]).shape[0]),
        'shape_len_indices_1': dataset_utils.int64_feature(np.array(dic["indices_r4_1"]).shape[0]),

        'radius_keypoints': dataset_utils.int64_feature(dic['radius_keypoints']),

    }))

    return example

##################################
#   Funzioni di Augumentation
##################################

def _aug_shift(dic_data, type, indx_img, tx=0, ty=0):
    if type == "or":
        assert (ty == 0)
    elif type == "ver":
        assert (tx == 0)

    h, w, c = dic_data["image_raw_"+str(indx_img)].shape

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dic_data["image_raw_"+str(indx_img)] = cv2.warpAffine(dic_data["image_raw_"+str(indx_img)], M, (w, h), flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_REPLICATE).reshape(h, w, c)

    dic_data["pose_mask_r4_"+str(indx_img)] = cv2.warpAffine(dic_data["pose_mask_r4_"+str(indx_img)], M, (w, h), flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_REPLICATE).reshape(h, w, c)

    keypoints_shifted = []
    values_shifted = []
    for coordinates in dic_data["indices_r4_"+str(indx_img)]:
        y, x, id = coordinates

        if type == "or":
            xs = x + tx
            ys = y
            if xs > 0 and xs < w:
                keypoints_shifted.append([ys, xs, id])
                values_shifted.append(1)

        elif type == "ver":
            xs = x
            ys = y + ty
            if ys > 0 and ys < h:
                keypoints_shifted.append([ys, xs, id])
                values_shifted.append(1)

    dic_data["indices_r4_"+str(indx_img)] = keypoints_shifted
    dic_data["values_r4_"+str(indx_img)] = values_shifted

    return dic_data

# luminoità tra max//3 e -max//3
def random_brightness(dic_data, indx_img):
    image = dic_data["image_raw_"+str(indx_img)]
    max = np.int64(image.max())
    max_d = max // 3
    min_d = -max_d
    brightness = tf.random.uniform(shape=[1], minval=min_d, maxval=max_d, dtype=tf.dtypes.int64).numpy()

    new_image = image + brightness
    new_image = np.clip(new_image, 0, 32765)
    new_image = new_image.astype(np.uint16)
    dic_data["image_raw_"+str(indx_img)] = new_image

    return dic_data

# contrasto tra max// e -max//2
def random_contrast(dic_data, indx_img):
    image = dic_data["image_raw_"+str(indx_img)]
    max = np.int64(image.max())
    max_d = max // 2
    min_d = -max_d
    contrast = tf.random.uniform(shape=[1], minval=min_d, maxval=max_d, dtype=tf.dtypes.int64).numpy()

    f = (max+4) * (contrast + max ) / (max * ((max+4) - contrast))
    alpha = f
    gamma = (max_d) * (1 - f)

    new_image = (alpha * image) + gamma
    new_image = np.clip(new_image, 0, 32765)
    new_image = new_image.astype(np.uint16)
    dic_data["image_raw_"+str(indx_img)] = new_image

    return dic_data

def _aug_rotation_angle(dic_data, angle_deegre, indx_img):

    h, w, c = dic_data["image_raw_"+str(indx_img)].shape
    ym, xm = h // 2, w // 2  # midpoint dell'immagine 96x128
    angle_radias = math.radians(angle_deegre)  # angolo di rotazione

    def rotate_keypoints(indices):
        keypoints_rotated = []
        for coordinates in indices:
            x, y = coordinates
            if y != -1 and x != -1:
                xr = (x - xm) * math.cos(angle_radias) - (y - ym) * math.sin(angle_radias) + xm
                yr = (x - xm) * math.sin(angle_radias) + (y - ym) * math.cos(angle_radias) + ym
                keypoints_rotated.append([int(xr), int(yr)])
            else:
                keypoints_rotated.append([x, y])

        return keypoints_rotated

    #If the angle is positive, the image gets rotated in the counter-clockwise direction.
    M = cv2.getRotationMatrix2D((xm, ym), -angle_deegre, 1.0)
    # Rotate image
    dic_data["image_raw_"+str(indx_img)] = cv2.warpAffine(dic_data["image_raw_"+str(indx_img)], M, (w, h),
                                             flags= cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_REPLICATE).reshape(h,w,c)

    # Rotate mask
    dic_data["pose_mask_r4_"+str(indx_img)] = cv2.warpAffine(dic_data["pose_mask_r4_"+str(indx_img)], M, (w, h)).reshape(h,w,c)

    # Rotate keypoints coordinate
    keypoints_rotated = rotate_keypoints(dic_data["original_peaks_"+str(indx_img)])
    dic_data["indices_r4_"+str(indx_img)], dic_data["values_r4_"+str(indx_img)] = _getSparsePose(keypoints_rotated, h, w, radius=dic_data['radius_keypoints'], mode='Solid')


    return dic_data

def _aug_flip(dic_data):

    ### Flip vertical pz_0
    mapping = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 10: 11, 11: 10, 9: 12, 12: 9, 8: 13, 13: 8}
    dic_data["image_raw_0"] = cv2.flip(dic_data["image_raw_0"], 1)
    dic_data["indices_r4_0"] = [[i[0], 64 + (64 - i[1]), mapping[i[2]]] for i in dic_data["indices_r4_0"]]
    dic_data["pose_mask_r4_0"] = cv2.flip(dic_data["pose_mask_r4_0"], 1)

    ### Flip vertical pz_1
    mapping = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 10: 11, 11: 10, 9: 12, 12: 9, 8: 13, 13: 8}
    dic_data["image_raw_1"] = cv2.flip(dic_data["image_raw_1"], 1)
    dic_data["indices_r4_1"] = [[i[0], 64 + (64 - i[1]), mapping[i[2]]] for i in dic_data["indices_r4_1"]]
    dic_data["pose_mask_r4_1"] = cv2.flip(dic_data["pose_mask_r4_1"], 1)

    return dic_data

###################################################################

def apply_augumentation(unprocess_dataset_it, config, type):

    sys.stdout.write('\n')
    sys.stdout.write("Applico augumentazione {type}..".format(type=type))
    sys.stdout.write('\n')

    num_example = None
    if type == "train":
        num_example = config.dataset_train_len
    elif type == "valid":
        num_example = config.dataset_valid_len
    elif type == "test":
        num_example = config.dataset_test_len

    name_tfrecord = type+'_augumentation.tfrecord'
    cnt_dataset = 0
    output_filename = os.path.join(config.data_tfrecord_path, name_tfrecord)
    tfrecord_writer = tf.compat.v1.python_io.TFRecordWriter(output_filename)

    for id_batch in range(num_example):#num example
        sys.stdout.write('\r')
        sys.stdout.write('Example: {id} / {tot}'.format(id=id_batch+1,tot=num_example))

        batch = next(unprocess_dataset_it)
        image_raw_0 = batch[0]  # [batch, 96, 128, 1]
        image_raw_1 = batch[1]  # [batch, 96,128, 1]
        mask_0 = batch[4]  # [batch, 96,128, 1]
        mask_1 = batch[5]  # [batch, 96,128, 1]
        pz_0 = batch[6]  # [batch, 1]
        pz_1 = batch[7]  # [batch, 1]
        name_0 = batch[8]  # [batch, 1]
        name_1 = batch[9]  # [batch, 1]
        indices_0 = batch[10]  # [batch, ]
        indices_1 = batch[11]  # [batch, ]
        values_0 = batch[12]  # [batch, ]
        values_1 = batch[13]  # [batch, ]
        original_peaks_0 = batch[14]
        original_peaks_1 = batch[15]
        radius_keypoints = batch[16]

        dic_data = {

            'pz_0': pz_0.numpy()[0].decode('utf-8'),  # nome del pz
            'pz_1': pz_1.numpy()[0].decode('utf-8'),

            'image_name_0': name_0.numpy()[0].decode('utf-8'),  # nome dell immagine 0
            'image_name_1': name_1.numpy()[0].decode('utf-8'),  # nome dell immagine 1
            'image_raw_0': image_raw_0.numpy()[0],  # immagine 0 in bytes
            'image_raw_1': image_raw_1.numpy()[0],  # immagine 1 in bytes

            'original_peaks_0': original_peaks_0.numpy()[0],
            'original_peaks_1': original_peaks_1.numpy()[0],

            'pose_mask_r4_0': mask_0.numpy()[0].astype(np.uint8),
            # maschera binaria a radius 4 con shape [96, 128, 1]
            'pose_mask_r4_1': mask_1.numpy()[0].astype(np.uint8),
            # maschera binaria a radius 4 con shape [96, 128, 1]

            'indices_r4_0': indices_0.numpy()[0],
            # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
            'values_r4_0': values_0.numpy()[0],
            # coordinate a radius 4 dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
            'indices_r4_1': indices_1.numpy()[0],
            # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 1, servono per ricostruire il vettore di sparse [num_indices, 3]
            'values_r4_1': values_1.numpy()[0],

            'radius_keypoints': radius_keypoints.numpy()[0]

        }
        example = _format_example(dic_data)
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
            dic_data_rotate = _aug_rotation_angle(dic_data.copy(), angle, indx_img=1) #rotazione image raw_1
            vec_dic_affine.append(dic_data_rotate)

        # Shift Random
        # Shift or
        random_shift_or_1 = tf.random.uniform(shape=[2], minval=-31, maxval=31, dtype=tf.int64).numpy()
        for shift in random_shift_or_1:
            dic_data_shifted = _aug_shift(dic_data.copy(), indx_img=1, type="or", tx=shift)
            vec_dic_affine.append(dic_data_shifted)
        # Shift ver
        random_shift_ver_1 = tf.random.uniform(shape=[2], minval=-11, maxval=11, dtype=tf.int64).numpy()
        for shift in random_shift_ver_1:
            dic_data_shifted = _aug_shift(dic_data.copy(), indx_img=1, type="ver", ty=shift)
            vec_dic_affine.append(dic_data_shifted)

        ### Aug image_raw_0
        #Piccole trasformaizoni
        list = vec_dic_affine.copy()
        for i, dic in enumerate(list):  # escludo l'immagine originale
            trasformation = tf.random.uniform(shape=[1], minval=0, maxval=4, dtype=tf.int64).numpy()
            if trasformation == 0:  # Nessuna trasformazione
                continue
            if trasformation == 1:  # Rotation
                angle = tf.random.uniform(shape=[1], minval=-46, maxval=46, dtype=tf.int64).numpy()[0]
                dic_new = _aug_rotation_angle(dic.copy(), angle, indx_img=0)  # rotazione image raw_1
                vec_dic_affine[i] = dic_new
            if trasformation == 2:  # Shift Or
                shift = tf.random.uniform(shape=[1], minval=-21, maxval=21, dtype=tf.int64).numpy()[0]
                dic_new = _aug_shift(dic.copy(), indx_img=0, type="or", tx=shift)  # rotazione image raw_1
                vec_dic_affine[i] = dic_new
            if trasformation == 3:  # Shift Ver
                shift = tf.random.uniform(shape=[1], minval=-11, maxval=11, dtype=tf.int64).numpy()[0]
                dic_new = _aug_shift(dic.copy(), indx_img=0, type="ver", ty=shift)  # rotazione image raw_1
                vec_dic_affine[i] = dic_new

        ### Salvo Affine trasformation
        for dic in vec_dic_affine:
            example = _format_example(dic)
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
            dic_data_rotate = _aug_rotation_angle(dic_data.copy(), angle, indx_img=1)  # rotazione image raw_1
            vec_dic_new_affine.append(dic_data_rotate)

        # Shift Random
        # Shift or
        random_shift_or_1 = tf.random.uniform(shape=[2], minval=-31, maxval=31, dtype=tf.int64).numpy()
        for shift in random_shift_or_1:
            dic_data_shifted = _aug_shift(dic_data.copy(), indx_img=1, type="or", tx=shift)
            vec_dic_new_affine.append(dic_data_shifted)
        # Shift ver
        random_shift_ver_1 = tf.random.uniform(shape=[2], minval=-11, maxval=11, dtype=tf.int64).numpy()
        for shift in random_shift_ver_1:
            dic_data_shifted = _aug_shift(dic_data.copy(), indx_img=1, type="ver", ty=shift)
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
                dic_new = _aug_rotation_angle(dic.copy(), angle, indx_img=0)  # rotazione image raw_1
                vec_dic_new_affine[i] = dic_new
            if trasformation == 2:  # Shift Or
                shift = tf.random.uniform(shape=[1], minval=-21, maxval=21, dtype=tf.int64).numpy()[0]
                dic_new = _aug_shift(dic.copy(), indx_img=0, type="or", tx=shift)  # rotazione image raw_1
                vec_dic_new_affine[i] = dic_new
            if trasformation == 3:  # Shift Ver
                shift = tf.random.uniform(shape=[1], minval=-11, maxval=11, dtype=tf.int64).numpy()[0]
                dic_new = _aug_shift(dic.copy(), indx_img=0, type="ver", ty=shift)  # rotazione image raw_1
                vec_dic_new_affine[i] = dic_new

        ############# Image raw 1 (structural)
        # Random B
        for dic in vec_dic_new_affine:
            dic_aug = random_brightness(dic.copy(), indx_img=0)
            vec_dic_structural.append(dic_aug)

        # Random C
        for dic in vec_dic_new_affine:
            dic_aug = random_contrast(dic.copy(), indx_img=0)
            vec_dic_structural.append(dic_aug)

        ###### Image raw 0 (structural)

        for i,dic in enumerate(vec_dic_structural):
            trasformation = tf.random.uniform(shape=[1], minval=0, maxval=3, dtype=tf.int64).numpy()
            if trasformation == 0:  # Nessuna trasformazione
                continue
            if trasformation == 1:  # brightness
                dic_new = random_brightness(dic.copy(), indx_img=0)
                vec_dic_structural[i] = dic_new
            if trasformation == 2:  # Contrast
                dic_new = random_contrast(dic.copy(), indx_img=0)
                vec_dic_structural[i] = dic_new


        ### Salvo Structural trasformation
        for dic in vec_dic_structural:
            example = _format_example(dic)
            tfrecord_writer.write(example.SerializeToString())
            cnt_dataset += 1


        ###############################
        # Flipping trasformation
        ###############################
        # Flip
        vec_tot_trasformation = vec_dic_affine + vec_dic_structural + [dic_data]
        for dic in vec_tot_trasformation:
            dic_aug = _aug_flip(dic.copy())
            example = _format_example(dic_aug)
            tfrecord_writer.write(example.SerializeToString())
            cnt_dataset += 1


    return name_tfrecord, cnt_dataset
