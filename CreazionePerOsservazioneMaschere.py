"""
Nel progetto originale prende il nomer di convert_market.py
Questo script consente di creare uno o più TFrecord che saranno utilizzati per il training del modello.
Nel TFrecord avremo le informazioni espresse in example del metodo _format_data
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


def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = np.sqrt(float(i ** 2 + j ** 2))
            if r + i >= 0 and r + i < height and c + j >= 0 and c + j < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([r + i, c + j, k])
                    values.append(1)
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('SparseKeypoint.png', dense * 255)
                elif 'Gaussian' == mode and distance <= radius:
                    indices.append([r + i, c + j, k])
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
shape -->  lista [height, width, channel]
"""


def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]  # coordinate peak ex: "300,200" type string
        c = int(p.split(',')[0])  # column
        r = int(p.split(',')[1])  # row
        if c != -1 and r != -1:  # non considero le occlusioni indicate con -1
            ind, val = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
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
            cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1)
            cv2.imwrite('keypoint.png', img)

    print("Log: Immagine salvata Keypoint")


"""
Creo le maschere binarie

@:return
Maschera con shape [height, width, 1]
"""


def _getPoseMask(peaks, height, width, radius_head=30, radius=4, var=4, mode='Solid'):
    limbSeq = [[0, 3], [0, 4], [0, 5],  # testa
               [1, 2], [2, 3],  # braccio dx
               [3, 4], [4, 5],  # collo
               [5, 6], [6, 7],  # braccio sx
               [11, 12], [12, 13],  # gamba sinistra
               [10, 9], [9, 8],  # gamba destra
               [11, 10], # Anche
               #Corpo
               [10, 4], [10, 3], [10, 5], [10, 0],
               [11, 4], [11, 3], [11, 5], [11, 0],
               ]
    indices = []
    values = []
    logs_cancellazione = [] # serve per controllare se entriamo negli if in cui manca o l'anca sx o dx o la spalla sx o dx
                  # len 0 --> non entro
                  # len > 0  --> sono entrato almeno una volta

    for limb in limbSeq:  # ad esempio limb = [2, 3]
        p0 = peaks[limb[0]]  # ad esempio coordinate per il keypoint corrispondente con id 2
        p1 = peaks[limb[1]]  # ad esempio coordinate per il keypoint corrispondente con id 3

        c0 = int(p0.split(',')[0])  # coordinata y  per il punto p0   ex: "280,235"
        r0 = int(p0.split(',')[1])  # coordinata x  per il punto p0
        c1 = int(p1.split(',')[0])  # coordinata y  per il punto p1
        r1 = int(p1.split(',')[1])  # coordinata x  per il punto p1

        if (limb[0] == 3 and r0 == -1) or (limb[1] == 3 and r1 == -1):  #manca spalla dx
            logs_cancellazione.append(3)
            continue
        if (limb[0] == 5 and r0 == -1) or (limb[1] == 5 and r1 == -1):  #manca spalla sx
            logs_cancellazione.append(5)
            continue
        if (limb[0] == 11 and r0 == -1) or (limb[1] == 11 and r1 == -1):  #manca anca sx
            logs_cancellazione.append(11)
            continue
        if (limb[0] == 10 and r0 == -1) or (limb[1] == 10 and r1 == -1): #manca anca dx
            logs_cancellazione.append(10)
            continue

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
            sampleN = int(distance / radius)  # numero di punti, con raggio di 4, di cui ho bisogno per coprire la distanza tra i punti p0 e p1
            if sampleN > 1:
                for i in range(1, sampleN):  # per ognuno dei punti di cui ho bisogno
                    r = int(r0 + (r1 - r0) * i / sampleN)  # calcolo della coordinata y
                    c = int(c0 + (c1 - c0) * i / sampleN)  # calcolo della coordinata x
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)  ## ingrandisco il nuovo punto considerando un raggio di 4
                    indices.extend(ind)
                    values.extend(val)

                    ## per stampare il linking dei lembi
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('Linking'+str(limb)+'.png', dense * 255)
            ## stampo l immagine
            dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
            cv2.imwrite('Dense{limb}.png'.format(limb=limb), dense * 255)

    if len(logs_cancellazione) == 0:
        shape = [height, width, 1]
        ## Fill body
        dense = np.squeeze(_sparse2dense(indices, values, shape))
        dense = dilation(dense, square(15))
        dense = erosion(dense, square(5))
        #cv2.imwrite('DenseMask.png', dense * 255)
    else:
        dense = None # significa che che manca almeno un keypoint tra spalla dx o sx e/o anca dx o sx
        logs_cancellazione = set(logs_cancellazione)

    return dense, list(logs_cancellazione)

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
            cv2.putText(img, str(k), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)
            #cv2.imwrite('keypoint.png', img)

    return img

def _format_data(config, pz_0, annotations_0):
    # Creazione del TFrecord

    # Read the image info:
    img_path_0 = os.path.join(config.data_path, pz_0, annotations_0['image'])

    image_raw_0 = tf.io.read_file(img_path_0)  # immagine in bytes 0
    height, width, _ = tf.io.decode_png(image_raw_0, channels=3).shape

    ########################## Pose 16x8 & Pose coodinate (for 128x64(Solid) 128x64(Gaussian))##########################
    ## Pose 16x8
    w_unit = width / 8  # per il rescaling delle coordinate dei keypoint
    h_unit = height / 16  # per il rescaling delle coordinate dei keypoint
    pose_peaks_0 = np.zeros([16, 8, config.keypoint_num])  # dimensioni maschere 16x8 con 14 heatmap una per ogni keypoint
    ## Pose coodinate
    pose_peaks_0_rcv = np.zeros([config.keypoint_num, 3])  ## Row, Column, Visibility

    #### Pose 0
    peaks = annotations_0[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    # shape --> [height, width, config.keypoint_num]
    indices_r4_0, values_r4_0, shape = _getSparsePose(peaks, height, width, config.keypoint_num, radius=4, mode='Solid')
    pose_mask_r4_0, logs_keypoints_mancanti = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
    if len(logs_keypoints_mancanti) == 0:

        keypoints_img = visualizePeaks(peaks, tf.io.decode_png(image_raw_0, channels=3).numpy())

        # Salvataggio delle 14 heatmap
        for ii in range(len(peaks)):
            p = peaks[ii]
            if 0 != len(p):
                c = int(p.split(',')[0])  # column
                r = int(p.split(',')[1])  # row
                ## siccome la dimensione delle immagini di posa sono 16x8, mentre i peaks sono considerati
                ## sull immagine raw di dimensione 128x64, ho bisogno di scalare le coordinate dei songoli peak.
                ## una volta fatto, per quello specifico punto setto il valore 1
                pose_peaks_0[int(r / h_unit), int(c / w_unit), ii] = 1
                ## il vettore rcv mi permette di ricorda le coordinate (non riscalate) e il valore settato
                pose_peaks_0_rcv[ii][0] = r
                pose_peaks_0_rcv[ii][1] = c
                pose_peaks_0_rcv[ii][2] = 1



        example = tf.train.Example(features=tf.train.Features(feature={

            'pz_0': dataset_utils.bytes_feature(pz_0.encode('utf-8')),
            'image_name_0': dataset_utils.bytes_feature(annotations_0['image'].encode('utf-8')),  # nome dell immagine 0
            'image_raw_0': dataset_utils.bytes_feature(image_raw_0.numpy()),  # immagine in bytes  0
            'key': dataset_utils.int64_feature(keypoints_img[:,:,0].astype(np.int64).flatten().tolist()),  # immagine con su stampati i keypoints

            'image_height': dataset_utils.int64_feature(height),  # 128
            'image_width': dataset_utils.int64_feature(width),  # 64

            'pose_mask_r4_0': dataset_utils.int64_feature(pose_mask_r4_0.astype(np.int64).flatten().tolist()),
            # maschera binaria a radius 4 con shape [128,64,1]

        }))
    else: # significa che che manca almeno un keypoint tra spalla dx o sx e/o anca dx o sx
        example = None

    return example, logs_keypoints_mancanti


if __name__ == '__main__':
    Config_file = __import__('0_config_utils')
    config = Config_file.Config()

    # Lettura delle prime annotazioni --> ex:pz3
    for name_file_annotation_0 in sorted(os.listdir(config.data_annotations_path), key = lambda x : int(x.split('_')[1].split('.')[0].split('z')[1])):

        pz_0 = name_file_annotation_0.split('_')[1].split('.csv')[0]
        path_annotation_0 = os.path.join(config.data_annotations_path, name_file_annotation_0)
        pd_annotation_0 = pd.read_csv(path_annotation_0, delimiter=';')
        len_0 = pd_annotation_0.shape[0]
        cnt =0

        output_filename = './masks/BabyPose_{pz}.tfrecord'.format(pz=pz_0)
        tfrecord_writer = tf.compat.v1.python_io.TFRecordWriter(output_filename)
        print(pz_0)
        logs_cancellazione_immagini = []

        # Creazione del pair
        for _, row_0 in pd_annotation_0.iterrows():
            # Creazione del tfrecord

            sys.stdout.write('\r>> Converting image %d/%d' % (cnt,len_0))
            sys.stdout.flush()
            example, logs_keypoints_mancanti = _format_data(config, pz_0, row_0)
            if None == example:
                logs_cancellazione_immagini.append([row_0['image'],logs_keypoints_mancanti])
                continue
            tfrecord_writer.write(example.SerializeToString())
            cnt += 1

        tfrecord_writer.close()
        np.save('./masks/logs_cancellazione/{pz}.npy'.format(pz=pz_0),np.array(logs_cancellazione_immagini))
        #TODO salvare logs per cancellazione immagine