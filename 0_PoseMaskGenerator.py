"""
Questo script consente di creare i TFrecord (train/valid/test) che saranno utilizzati per il training del modello.
Per modificare le informazioni, fare riferimento alla sezione CONFIG presente nel main
"""

import math
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from random import randint
from skimage.morphology import square, dilation, erosion
from utils import dataset_utils
import pickle
import imutils

"""
# Data una shape [128, 64, 1] e dati come indices tutti i punti (sia i keypoint che la copertura tra loro),
# andiamo a creare una maschera binaria in cui solamente la sagoma della persona assuma valore 1
"""
def _sparse2dense(indices, values, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        dense[r, c, 0] = values[i]
    return dense

"""
# Dato un radius di 4 e un punto p ciò che cerco di fare è di trovare tutti i punti che si trovano 
# nell'intorno [-4,4] del punto p. Le coordinate di ognuno di questi punti le salvo in indices e setto il valore 1 (visibile).
# Al termine, ciò che otteniamo è che il punto p viene ingrandito considerando un raggio di 4.
# Un esempio è mostrato in figura SparseKeypoint.png
"""
def _getSparseKeypoint(y, x, k, height, width, radius=4, var=4, mode='Solid'):
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
                elif 'Gaussian' == mode and distance <= radius:
                    indices.append([x + j, y + i, k])
                    if 4 == var:
                        values.append(Gaussian_0_4.pdf(distance) * Ratio_0_4)
                    else:
                        assert 'Only define Ratio_0_4  Gaussian_0_4 ...'
    return indices, values


"""
# Andiamo ad ingrandire ogni peaks di un raggio 4 o superiore creando nuovi punti.
# Salviamo tutti in indices. I Values sono settati ad 1 ed indicano la visibilità degli indices
# i valori di k indicano gli indici di ogni keypoint:
  0 head; 1 right_hand; 2 right_elbow; 3 right_shoulder; 4 neck; 5 left_shoulder; 6 left_elbow;
  7 left_hand; 8 right_foot; 9 right_knee; 10 right_hip; 11 left_hip; 12 left_knee; 13 left_foot
  
@:return
indices --> [ [<coordinata>, <coordinata>, <indice keypoint>], ... ]
values --> [  1,1,1, ... ]
shape -->  lista [height, width, num keypoints]
"""
def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k] # coordinate peak ex: "300,200"
        x = p[0]
        y = p[1]
        if x != -1 and y != -1:  # non considero le occlusioni indicate con -1
            ind, val = _getSparseKeypoint(y, x, k, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
    shape = [height, width, channel]
    return indices, values, shape

"""
## Serve per stamapare i peaks sull immagine in input. Stampo i peaks considerando anche i corrispettivi id
"""
def visualizePeaks(peaks, img):
    for k in range(len(peaks)):
        p = peaks[k]  # coordinate peak ex: "300,200" type string
        x = int(p.split(',')[0])  # column
        y = int(p.split(',')[1])  # row
        if x != -1 and y != -1:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(img, str(k), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1)
            cv2.imwrite('keypoint.png', img)

    print("Log: Immagine salvata Keypoint")

"""
Creo le maschere binarie

@:return
Maschera con shape [height, width, 1]
"""
def _getPoseMask(peaks, height, width, radius_head=60, radius=4, dilatation= 10, var=4, mode='Solid'):
    limbSeq = [[0, 3], [0, 4], [0, 5],  # testa
               [1, 2], [2, 3],  # braccio dx
               [3, 4], [4, 5],  # collo
               [5, 6], [6, 7],  # braccio sx
               [11, 12], [12, 13],  # gamba sinistra
               [10, 9], [9, 8],  # gamba destra
               [11, 10],  # Anche
               # Corpo
               [10, 4], [10, 3], [10, 5], [10, 0],
               [11, 4], [11, 3], [11, 5], [11, 0],
               ]
    indices = []
    values = []
    for limb in limbSeq:  # ad esempio limb = [2, 3]
        p0 = peaks[limb[0]]  # ad esempio coordinate per il keypoint corrispondente con id 2
        p1 = peaks[limb[1]]  # ad esempio coordinate per il keypoint corrispondente con id 3

        c0 = int(p0.split(',')[0])  # coordinata y  per il punto p0   ex: "280,235"
        r0 = int(p0.split(',')[1])  # coordinata x  per il punto p0
        c1 = int(p1.split(',')[0])  # coordinata y  per il punto p1
        r1 = int(p1.split(',')[1])  # coordinata x  per il punto p1

        if r0 != -1 and r1 != -1 and c0 != -1 and c1 != -1:  # non considero le occlusioni che sono indicate con valore -1

            if limb[0] == 0:  # Per la testa utilizzo un Radius maggiore
                ind, val = _getSparseKeypoint(r0, c0, 0, height, width, radius_head, var,
                                              mode)  # ingrandisco il punto p0 considerando un raggio di 20
            else:
                ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var,
                                              mode)  # # ingrandisco il punto p1 considerando un raggio di 4

            indices.extend(ind)
            values.extend(val)

            if limb[1] == 0:
                # Per la testa utilizzo un Radius maggiore
                ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius_head, var,
                                              mode)  # ingrandisco il punto p1 considerando un raggio di 20
            else:
                ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var,
                                              mode)  # # ingrandisco il punto p1 considerando un raggio di 4

            indices.extend(ind)
            values.extend(val)

            # ## stampo l immagine
            # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
            # cv2.imwrite('Dense{limb}.png'.format(limb=limb), dense * 255)

            # Qui vado a riempire il segmento ad esempio [2,3] corrispondenti ai punti p0 e p1.
            # L idea è quella di riempire questo segmento con altri punti di larghezza radius=4.
            # Ovviamente per farlo calcolo la distanza tra p0 e p1 e poi vedo quanti punti di raggio 4 entrano in questa distanza.
            # Un esempio è mostrato nelle varie imamgini di linking
            distance = np.sqrt((r0 - r1) ** 2 + (c0 - c1) ** 2)  # distanza tra il punto p0 e p1
            sampleN = int(
                distance / radius)  # numero di punti, con raggio di 4, di cui ho bisogno per coprire la distanza tra i punti p0 e p1
            if sampleN > 1:
                for i in range(1, sampleN):  # per ognuno dei punti di cui ho bisogno
                    r = int(r0 + (r1 - r0) * i / sampleN)  # calcolo della coordinata y
                    c = int(c0 + (c1 - c0) * i / sampleN)  # calcolo della coordinata x
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var,
                                                  mode)  ## ingrandisco il nuovo punto considerando un raggio di 4
                    indices.extend(ind)
                    values.extend(val)

                    ## per stampare il linking dei lembi
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('Linking'+str(limb)+'.png', dense * 255)
            ## stampo l immagine
            dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
            #cv2.imwrite('Dense{limb}.png'.format(limb=limb), dense * 255)


    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    dense = dilation(dense, square(35))
    dense = erosion(dense, square(5))
    dense = np.reshape(dense, [dense.shape[0], dense.shape[1], 1])
    #cv2.imwrite('DenseMask.png', dense * 255)

    return dense


"""
Augumentation con flip verticale
@:return dic_data --> dic con dati augumentati
"""
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

"""
Augumentation con shift 
(tx, ty) --> punto in cui devo spostare il punto (0,0) in alto a sx dell immagine in input
@:return dic_data --> dic con dati augumentati
"""
def _aug_shift(dic_data, type, tx=0, ty=0):
    if type == "or":
        assert( ty == 0)
    elif type == "ver":
        assert( tx == 0)


    h, w, c = dic_data["image_raw_0"].shape

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dic_data["image_raw_1"] = cv2.warpAffine(dic_data["image_raw_1"], M, (w, h), flags= cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_REPLICATE).reshape(h,w,c)

    dic_data["pose_mask_r4_1"] = cv2.warpAffine(dic_data["pose_mask_r4_1"], M, (w, h), flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_REPLICATE).reshape(h, w, c)

    keypoints_shifted = []
    values_shifted = []
    for coordinates in dic_data["indices_r4_1"]:
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



    dic_data["indices_r4_1"] = keypoints_shifted
    dic_data["values_r4_1"] = values_shifted

    return dic_data


def random_brightness(dic_data):
    dic_data["image_raw_1"] = tf.keras.preprocessing.image.random_brightness(dic_data["image_raw_1"], (1.5,2.0)).astype(np.uint16)

    return dic_data

def random_contrast(dic_data):
    dic_data["image_raw_1"] = tf.image.random_contrast(dic_data["image_raw_1"], lower=0.2, upper=0.5, seed=2).numpy().astype(np.uint16)

    return dic_data


"""
Augumentation con rotazione secondo angolo angle
@:return dic_data --> dic con dati augumentati
"""
def _aug_rotation_angle(dic_data, angle_deegre):

    h, w, c = dic_data["image_raw_0"].shape
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

    M = cv2.getRotationMatrix2D((xm, ym), -angle_deegre, 1.0)
    # Rotate image
    dic_data["image_raw_1"] = cv2.warpAffine(dic_data["image_raw_1"], M, (w, h),
                                             flags= cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_REPLICATE).reshape(h,w,c)

    # Rotate mask
    dic_data["pose_mask_r4_1"] = cv2.warpAffine(dic_data["pose_mask_r4_1"], M, (w, h)).reshape(h,w,c)

    # Rotate keypoints coordinate
    keypoints_rotated = rotate_keypoints(dic_data["original_peaks_1"])
    dic_data["indices_r4_1"], dic_data["values_r4_1"], _ = _getSparsePose(keypoints_rotated, h, w, 14,  radius=radius_keypoints_pose, mode='Solid')


    return dic_data

"""
Crezione dell example da aggiungere al TF Record
@:return example
"""
def _format_example(dic):


    example = tf.train.Example(features=tf.train.Features(feature={

        'pz_0': dataset_utils.bytes_feature(dic["pz_0"].encode('utf-8')),  # nome del pz
        'pz_1': dataset_utils.bytes_feature(dic["pz_1"].encode('utf-8')),

        'image_name_0': dataset_utils.bytes_feature(dic["image_name_0"].encode('utf-8')),  # nome dell immagine 0
        'image_name_1': dataset_utils.bytes_feature(dic["image_name_0"].encode('utf-8')),  # nome dell immagine 1
        'image_raw_0': dataset_utils.bytes_feature(dic["image_raw_0"].tostring()),  # immagine 0 in bytes
        'image_raw_1': dataset_utils.bytes_feature(dic["image_raw_1"].tostring()),  # immagine 1 in bytes

        'image_format': dataset_utils.bytes_feature('PNG'.encode('utf-8')),
        'image_height': dataset_utils.int64_feature(96),
        'image_width': dataset_utils.int64_feature(128),

        "original_peaks_0": dataset_utils.bytes_feature(np.array(dic["original_peaks_0"]).astype(np.int64).tostring()),
        "original_peaks_1": dataset_utils.bytes_feature(np.array(dic["original_peaks_1"]).astype(np.int64).tostring()),
        'shape_len_original_peaks_0': dataset_utils.int64_feature(np.array(dic["original_peaks_0"]).shape[0]),
        'shape_len_original_peaks_1': dataset_utils.int64_feature(np.array(dic["original_peaks_1"]).shape[0]),

        'pose_mask_r4_0': dataset_utils.int64_feature(dic["pose_mask_r4_0"].astype(np.uint8).flatten().tolist()),
        # maschera binaria a radius 4 con shape [96, 128, 1]
        'pose_mask_r4_1': dataset_utils.int64_feature(dic["pose_mask_r4_1"].astype(np.uint8).flatten().tolist()),
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

        'radius_keypoints': dataset_utils.int64_feature(radius_keypoints_pose),

    }))

    return example

"""
Crezione dei dati 
@:return dic_data --> un dzionario in cui sono contenuti i dati creati img_0, img_1 etc..
"""
def _format_data( id_pz_0, id_pz_1, annotations_0, annotations_1,
                 radius_keypoints_pose, radius_keypoints_mask,
                 radius_head_mask, dilatation ):

    pz_0 = 'pz'+str(id_pz_0)
    pz_1 = 'pz' + str(id_pz_1)

    # Read the image info:
    name_img_0_16_bit = annotations_0['image'].split('_')[0] + '_16bit.png'
    name_img_1_16_bit = annotations_1['image'].split('_')[0] + '_16bit.png'
    img_path_0 = os.path.join(dir_data, pz_0, name_img_0_16_bit)
    img_path_1 = os.path.join(dir_data, pz_1, name_img_1_16_bit)

    # Read immagine
    image_0 = cv2.imread(img_path_0, cv2.IMREAD_UNCHANGED) #[480,640]
    image_1 = cv2.imread(img_path_1, cv2.IMREAD_UNCHANGED) #[480,640]
    height, width  = image_0.shape[0], image_0.shape[1]

    ### Pose coodinate

    ### Pose image 0 a radius 4
    peaks = annotations_0[1:] # annotation_0[1:] --> poichè tolgo il campo image
    pose_mask_r4_0 = _getPoseMask(peaks, height, width, radius=radius_keypoints_mask, radius_head=radius_head_mask,  dilatation=dilatation,
                                  mode='Solid') #[480,640,1]


    ### Resize a 96x128 pose 0
    image_0 = cv2.resize(image_0, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]
    peaks_resized_0 = []
    # resize dei peaks
    for k in range(len(peaks)):
        p = peaks[k]  # coordinate peak ex: "300,200" type string
        x = int(p.split(',')[0])  # column
        y = int(p.split(',')[1])  # row
        if x != -1 and y != -1:
            peaks_resized_0.append([int(x / 5), int(y / 5)])  # 5 è lo scale factor --> 480/96 e 640/128
        else:
            peaks_resized_0.append([x, y])
    indices_r4_0, values_r4_0, _ = _getSparsePose(peaks_resized_0, height, width, keypoint_num, radius=radius_keypoints_pose, mode='Solid')  # shape
    pose_mask_r4_0 = cv2.resize(pose_mask_r4_0, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]


    #### Pose 1 radius 4
    peaks = annotations_1[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=radius_keypoints_mask, radius_head=radius_head_mask,
                                  dilatation=dilatation,
                                  mode='Solid')

    ## Reshape a 96x128 pose 1
    image_1 = cv2.resize(image_1, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]
    peaks_resized_1 = []
    #resize dei peaks
    for k in range(len(peaks)):
        p = peaks[k] # coordinate peak ex: "300,200" type string
        x = int(p.split(',')[0])  # column
        y = int(p.split(',')[1])  # row
        if y != -1 and x != -1:
            peaks_resized_1.append([int(x / 5), int(y / 5)]) # 5 è lo scale factor --> 480/96 e 640/128
        else:
            peaks_resized_1.append([x ,y])
    indices_r4_1, values_r4_1, _ = _getSparsePose(peaks_resized_1, height, width, keypoint_num, radius=radius_keypoints_pose, mode='Solid')  # shape
    pose_mask_r4_1 = cv2.resize(pose_mask_r4_1, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]


    dic_data = {

        'pz_0': pz_0,  # nome del pz
        'pz_1': pz_1,

        'image_name_0': name_img_0_16_bit,  # nome dell immagine 0
        'image_name_1': name_img_1_16_bit,  # nome dell immagine 1
        'image_raw_0': image_0,  # immagine 0 in bytes
        'image_raw_1': image_1,  # immagine 1 in bytes

        'original_peaks_0': peaks_resized_0, #peaks ridimensionati a 96x128
        'original_peaks_1': peaks_resized_1,

        'pose_mask_r4_0': pose_mask_r4_0,
        # maschera binaria a radius 4 con shape [96, 128, 1]
        'pose_mask_r4_1': pose_mask_r4_1,
        # maschera binaria a radius 4 con shape [96, 128, 1]

        'indices_r4_0': indices_r4_0,
        # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'values_r4_0': values_r4_0,
        # coordinate a radius 4 dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'indices_r4_1': indices_r4_1,
        # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 1, servono per ricostruire il vettore di sparse [num_indices, 3]
        'values_r4_1': values_r4_1

    }

    return dic_data

"""
Controllo se l'immagine contiene i Keypoint 3,5,10,11 rispettivamente di spalla dx e sx e anca dx e sx
@:return True --> almeno uno dei keypoint è mancanta
@:return False --> i keypoints ci sono tutti
"""
def check_assenza_keypoints_3_5_10_11(peaks):

    if int(peaks[3].split(',')[0]) == -1 or int(peaks[5].split(',')[0]) == -1 or int(peaks[10].split(',')[0]) == -1 or int(peaks[11].split(',')[0]) == -1:
        return True
    else:
        return False


"""
Consente di selezionare la coppia di pair da formare
"""
def fill_tfrecord(lista, tfrecord_writer, radius_keypoints_pose, radius_keypoints_mask,
                radius_head_mask, dilatation,campionamento, flip=False, mode="negative", switch=False):

    tot_pairs = 0 # serve per contare il totale di pair nel tfrecord

    # Accoppiamento tra immagini appartenenti allo stesso pz
    if mode == "positive":
        for pz_0 in lista:
            pz_1 = pz_0
            name_file_annotation_0 = 'result_pz{id}.csv'.format(id=pz_0)
            path_annotation_0 = os.path.join(dir_annotations, name_file_annotation_0)
            df_annotation_0 = pd.read_csv(path_annotation_0, delimiter=';')

            cnt = 0  # Serve per printare a schermo il numero di example. Lo resettiamo ad uno ad ogni nuovo pz_1

            # Creazione del pair
            for i in range(0 , len(df_annotation_0)):
                row_0 = df_annotation_0.loc[i]

                # Controllo se l'immagine row_0 contiene i keypoints relativi alla spalla dx e sx e anca dx e sx
                # In caso di assenza passo all'immagine successiva
                if check_assenza_keypoints_3_5_10_11(row_0[1:]):
                    continue

                for j in range(i + 1, len(df_annotation_0)):
                    row_1 = df_annotation_0.loc[j]

                    # Controllo se l'immagine row_0 contiene i keypoints relativi alla spalla dx e sx e anca dx e sx
                    # In caso di assenza passo all'immagine successiva
                    if check_assenza_keypoints_3_5_10_11(row_1[1:]):
                        continue

                    # Creazione dell'example tfrecord
                    dic_data = _format_data(pz_0, pz_1, row_0, row_1, radius_keypoints_pose, radius_keypoints_mask,
                                            radius_head_mask, dilatation)
                    example = _format_example(dic_data)
                    tfrecord_writer.write(example.SerializeToString())
                    cnt += 1  # incremento del conteggio degli examples
                    tot_pairs += 1

                    if switch:
                        dic_data_switch = _format_data(pz_1, pz_0, row_1, row_0,radius_keypoints_pose, radius_keypoints_mask,
                radius_head_mask, dilatation,)
                        example = _format_example(dic_data_switch)
                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1  # incremento del conteggio degli examples
                        tot_pairs += 1

                    if flip:
                        dic_data_flip = _aug_flip(dic_data.copy())
                        example = _format_example(dic_data_flip)
                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1  # incremento del conteggio degli examples
                        tot_pairs += 1
                #         if switch:
                #             dic_data_switch_flip = _aug_flip(dic_data_switch.copy())
                #             example = _format_example(dic_data_switch_flip)
                #             tfrecord_writer.write(example.SerializeToString())
                #             cnt += 1  # incremento del conteggio degli examples
                #             tot_pairs += 1


                    sys.stdout.write(
                        '\r>> Creazione pair [{pz_0}, {pz_1}] image {cnt}/{tot}'.format(pz_0=pz_0, pz_1=pz_1, cnt=cnt,
                                                                                        tot=df_annotation_0.shape[0]))
                    sys.stdout.flush()

    # Accoppiamento tra immagini appartenenti a pz differenti
    if mode == "negative":
        # Lettura delle prime annotazioni --> ex:pz3
        for pz_0 in lista:
            name_file_annotation_0 = 'result_pz{id}.csv'.format(id=pz_0)
            path_annotation_0 = os.path.join(dir_annotations, name_file_annotation_0)
            df_annotation_0 = pd.read_csv(path_annotation_0, delimiter=';')

            # Lettura delle seconde annotazioni --> ex:pz4
            for pz_1 in lista:
                if pz_0 != pz_1:
                    name_file_annotation_1 = 'result_pz{id}.csv'.format(id=pz_1)
                    name_path_annotation_1 = os.path.join(dir_annotations, name_file_annotation_1)
                    df_annotation_1 = pd.read_csv(name_path_annotation_1, delimiter=';')

                    cnt = 0 # Serve per printare a schermo il numero di example. Lo resettiamo ad uno ad ogni nuovo pz_1

                    # Creazione del pair
                    for indx, row_0 in df_annotation_0.iterrows():

                            # Controllo se l'immagine row_0 contiene i keypoints relativi alla spalla dx e sx e anca dx e sx
                            # In caso di assenza passo all'immagine successiva
                            # Controllo se la row_0 è nelle immagini selezionate
                            if check_assenza_keypoints_3_5_10_11(row_0[1:]):
                                continue

                            
                            # lettura random delle row_1 nel secondo dataframe
                            value = randint(0, len(df_annotation_1) - 1)
                            row_1 = df_annotation_1.loc[value]
                           
                            
                            # Controllo se l'immagine row_1 contiene i keypoints relativi alla spalla dx e sx e anca dx e sx
                            # In caso di assenza passo ne seleziona un altra random
                            conteggio_while = 0
                            while check_assenza_keypoints_3_5_10_11(row_1[1:]):
                                if conteggio_while < 20:
                                    # lettura random delle row_1 nel secondo dataframe
                                    value = randint(0, len(df_annotation_1) - 1)
                                    row_1 = df_annotation_1.loc[value]
                                    conteggio_while += 1
                                else:
                                    r = input("Sono entrato più di 20 volte nel while")

                            #row_1 = df_annotation_1.loc[indx]

                            # Creazione dell'example tfrecord
                            dic_data = _format_data(pz_0, pz_1, row_0, row_1,radius_keypoints_pose, radius_keypoints_mask,
                    radius_head_mask, dilatation)
                            example = _format_example(dic_data)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1

                            if flip:
                                dic_data_flip = _aug_flip(dic_data.copy())
                                example = _format_example(dic_data_flip)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1

                            """
                            dic_data_rotate = _aug_rotation_angle(dic_data.copy(), 45)
                            example = _format_example(dic_data_rotate)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_rotate = _aug_flip(dic_data_rotate.copy())
                                example = _format_example(dic_data_rotate)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_rotate = _aug_rotation_angle(dic_data.copy(), 315)
                            example = _format_example(dic_data_rotate)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_rotate = _aug_flip(dic_data_rotate.copy())
                                example = _format_example(dic_data_rotate)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
    
                            dic_data_rotate = _aug_rotation_angle(dic_data.copy(), 90)
                            example = _format_example(dic_data_rotate)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_rotate = _aug_flip(dic_data_rotate.copy())
                                example = _format_example(dic_data_rotate)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_rotate = _aug_rotation_angle(dic_data.copy(), -90)
                            example = _format_example(dic_data_rotate)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_rotate = _aug_flip(dic_data_rotate.copy())
                                example = _format_example(dic_data_rotate)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_shifted = _aug_shift(dic_data.copy(), type="or", tx=30)
                            example = _format_example(dic_data_shifted)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_shifted = _aug_flip(dic_data_shifted.copy())
                                example = _format_example(dic_data_shifted)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_shifted = _aug_shift(dic_data.copy(), type="or", tx=-30)
                            example = _format_example(dic_data_shifted)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_shifted = _aug_flip(dic_data_shifted.copy())
                                example = _format_example(dic_data_shifted)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_shifted = _aug_shift(dic_data.copy(), type="ver", ty=10)
                            example = _format_example(dic_data_shifted)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_shifted = _aug_flip(dic_data_shifted.copy())
                                example = _format_example(dic_data_shifted)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_shifted = _aug_shift(dic_data.copy(), type="ver", ty=-10)
                            example = _format_example(dic_data_shifted)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            if flip:
                                dic_data_shifted = _aug_flip(dic_data_shifted.copy())
                                example = _format_example(dic_data_shifted)
                                tfrecord_writer.write(example.SerializeToString())
                                cnt += 1  # incremento del conteggio degli examples
                                tot_pairs += 1
    
                            dic_data_random_b = random_brightness(dic_data.copy())
                            example = _format_example(dic_data_random_b)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
    
                            dic_data_random_c = random_contrast(dic_data.copy())
                            example = _format_example(dic_data_random_c)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1
                            """


                            sys.stdout.write(
                                '\r>> Creazione pair [{pz_0}, {pz_1}] image {cnt}/{tot}'.format(pz_0=pz_0, pz_1=pz_1,
                                                                                                cnt=cnt,
                                                                                                tot= df_annotation_0.shape[0]))
                            sys.stdout.flush()

                    sys.stdout.write('\n')
                    sys.stdout.flush()

            sys.stdout.write('\nTerminato {pz_0}'.format(pz_0=pz_0))
            sys.stdout.write('\n\n')
            sys.stdout.flush()

    tfrecord_writer.close()
    sys.stdout.write('\nSET DATI TERMINATO')
    sys.stdout.write('\n\n')
    sys.stdout.flush()


    return tot_pairs


if __name__ == '__main__':

#### CONFIG ##########

    global dir_save_tfrecord
    global dir_annotations
    global dir_data
    global keypoint_num

    dir_data = './data/Syntetich'
    dir_annotations = './data/Syntetich/annotations'
    dir_save_tfrecord = './data/Syntetich/tfrecord/negative_no_flip'
    keypoint_num = 14

    name_tfrecord_train = 'BabyPose_train.tfrecord'
    name_tfrecord_valid = 'BabyPose_valid.tfrecord'
    name_tfrecord_test = 'BabyPose_test.tfrecord'

    # liste contenente i num dei pz che vanno considerati per singolo set
    lista_pz_train = [101, 103, 105, 106, 107, 109, 110, 112]
    lista_pz_valid = [102, 111]
    lista_pz_test = [104, 108]

    # General information
    radius_keypoints_pose = 1
    radius_keypoints_mask = 2
    radius_head_mask = 40
    dilatation = 35
    campionamento = 0
    flip = False # Aggiunta dell example con flip verticale
    mode = "negative"
    switch = mode == "positive" #lo switch è consentito solamente in modalità positive, se è negative va in automatico

#########################

    # Create file tfrecord
    output_filename_train = os.path.join(dir_save_tfrecord, name_tfrecord_train)
    output_filename_valid = os.path.join(dir_save_tfrecord, name_tfrecord_valid)
    output_filename_test = os.path.join(dir_save_tfrecord, name_tfrecord_test)

    r_tr = None
    r_v = None
    r_te = None

    if os.path.exists(output_filename_train):
        r_tr = input("Il tf record di train esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_tr == "Y" or r_tr == "N" or r_tr == "y" or r_tr == "n"
    if not os.path.exists(output_filename_train) or r_tr == "Y" or r_tr == "y":
        tfrecord_writer_train = tf.compat.v1.python_io.TFRecordWriter(output_filename_train)
        tot_train = fill_tfrecord(lista_pz_train, tfrecord_writer_train, radius_keypoints_pose, radius_keypoints_mask,
                                  radius_head_mask, dilatation, campionamento, flip=flip, mode=mode, switch=switch)
        print("TOT TRAIN: ", tot_train)
    elif r_tr == "N" or r_tr == "n":
        print("OK, non farò nulla sul train set")

    if os.path.exists(output_filename_valid):
        r_v = input("Il tf record di valid esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_v == "Y" or r_v == "N" or r_v == "y" or r_v == "n"
    if not os.path.exists(output_filename_valid) or r_v == "Y" or r_v == "y":
        tfrecord_writer_valid = tf.compat.v1.python_io.TFRecordWriter(output_filename_valid)
        tot_valid = fill_tfrecord(lista_pz_valid, tfrecord_writer_valid, radius_keypoints_pose, radius_keypoints_mask,
                                  radius_head_mask, dilatation,campionamento, flip=flip, mode=mode, switch=switch)
        print("TOT VALID: ", tot_valid)
    elif r_v == "N" or r_v == "n":
        print("OK, non farò nulla sul valid set")

    if os.path.exists(output_filename_test):
        r_te = input("Il tf record di test esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_te == "Y" or r_te == "N" or r_te == "y" or r_te == "n"
    if not os.path.exists(output_filename_test) or r_te == "Y" or r_te == "y":
        tfrecord_writer_test = tf.compat.v1.python_io.TFRecordWriter(output_filename_test)
        tot_test = fill_tfrecord(lista_pz_test, tfrecord_writer_test, radius_keypoints_pose, radius_keypoints_mask,
                                  radius_head_mask, dilatation, campionamento, flip=flip, mode=mode)
        print("TOT TEST: ", tot_test)
    elif r_te == "N" or r_te == "n":
        print("OK, non farò nulla sul test set")

    dic = {

        "general": {
            "radius_keypoints_pose": radius_keypoints_pose,
            "radius_keypoints_mask": radius_keypoints_mask,
            "radius_head_mask": radius_head_mask,
            "dilatation": dilatation,
            "flip": flip,
            "mode": mode
        },

        "train": {
        "name_file": name_tfrecord_train,
        "list_pz": lista_pz_train,
        "tot": tot_train
    },
        "valid": {
            "name_file": name_tfrecord_valid,
            "list_pz": lista_pz_valid,
            "tot": tot_valid
        },

        "test": {
            "name_file": name_tfrecord_test,
            "list_pz": lista_pz_test,
            "tot": tot_test
        }
    }
    log_tot_sets = os.path.join(dir_save_tfrecord, 'pair_tot_sets.pkl')
    f = open(log_tot_sets, "wb")
    pickle.dump(dic, f)
    f.close()




