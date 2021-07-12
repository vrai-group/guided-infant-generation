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
import pickle

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
shape -->  lista [height, width, num keypoints]
"""
def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k] # coordinate peak ex: "300,200" type string
        c = p[0] # column
        r = p[1]  # row
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
            cv2.putText(img, str(k), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1)
            cv2.imwrite('keypoint.png', img)

    print("Log: Immagine salvata Keypoint")

"""
Creo le maschere binarie

@:return
Maschera con shape [height, width, 1]
"""
def _getPoseMask(peaks, height, width, radius_head=60, radius=4, var=4, mode='Solid'):
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
Crezione dell example da aggiungere al TF Record
@:return example
"""
def _format_data(config, id_pz_0, id_pz_1, annotations_0, annotations_1):

    pz_0 = 'pz'+str(id_pz_0)
    pz_1 = 'pz' + str(id_pz_1)

    # Read the image info:
    name_img_0_16_bit = annotations_0['image'].split('_')[0] + '_16bit.png'
    name_img_1_16_bit = annotations_1['image'].split('_')[0] + '_16bit.png'
    img_path_0 = os.path.join(config.data_path, pz_0, name_img_0_16_bit)
    img_path_1 = os.path.join(config.data_path, pz_1, name_img_1_16_bit)

    # Read immagine
    image_0 = cv2.imread(img_path_0, cv2.IMREAD_UNCHANGED) #[480,640]
    image_1 = cv2.imread(img_path_1, cv2.IMREAD_UNCHANGED) #[480,640]
    height, width  = image_0.shape[0], image_0.shape[1]

    ### Pose 8x16
    # Qui salviamo i peaks iniziali (quelli che abbiamo nei cvs annotations) sottoforma di maschere di dimensioni 16x8
    w_unit = width / 16  # per il rescaling delle coordinate dei keypoint  per immagine a 8*16
    h_unit = height / 8  # per il rescaling delle coordinate dei keypoint per immagine a 8*16
    pose_peaks_0 = np.zeros([8, 16, config.keypoint_num])
    pose_peaks_1 = np.zeros([8, 16, config.keypoint_num])

    ### Pose coodinate
    # Qui salviamo le coordinate dei keypoint iniziali (quelli che abbiamo nei csv annotations) in un array che contiene tre informaioni:
    # coordinata Row del peak, coordinata Column del peek, Visibility del peak
    pose_peaks_0_rcv = np.zeros([config.keypoint_num, 3])
    pose_peaks_1_rcv = np.zeros([config.keypoint_num, 3])

    ### Pose image 0 a radius 4
    peaks = annotations_0[1:] # annotation_0[1:] --> poichè tolgo il campo image
    pose_mask_r4_0 = _getPoseMask(peaks, height, width, radius=10, radius_head=60,  mode='Solid') #[480,640,1]


    ### Resize a 96x128 pose 0
    image_0 = cv2.resize(image_0, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]
    peaks_resized = []
    # resize dei peaks
    for k in range(len(peaks)):
        p = peaks[k]  # coordinate peak ex: "300,200" type string
        c = int(p.split(',')[0])  # column
        r = int(p.split(',')[1])  # row
        if c != -1 and r != -1:
            peaks_resized.append([int(c / 5), int(r / 5)])  # 5 è lo scale factor --> 480/96 e 640/128
        else:
            peaks_resized.append([c, r])
    indices_r4_0, values_r4_0, _ = _getSparsePose(peaks_resized, height, width, config.keypoint_num, radius=2, mode='Solid')  # shape
    pose_mask_r4_0 = cv2.resize(pose_mask_r4_0, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]


    # Salvataggio delle 14 heatmap a 8x16 e dei vettori coordinata
    for ii in range(len(peaks)):
        p = peaks[ii]
        if 0 != len(p):
            c = int(p.split(',')[0])  # column
            r = int(p.split(',')[1])  # row
            # siccome la dimensione delle immagini di posa sono 8/16, mentre i peaks sono considerati
            # sull immagine raw di dimensione 480x640, ho bisogno di scalare le coordinate dei songoli peak.
            # una volta fatto, per quello specifico punto setto il valore visibility=1
            pose_peaks_0[int(r / h_unit), int(c / w_unit), ii] = 1
            # il vettore rcv mi permette di ricorda le coordinate (non riscalate) e il valore settato
            pose_peaks_0_rcv[ii][0] = r
            pose_peaks_0_rcv[ii][1] = c
            pose_peaks_0_rcv[ii][2] = 1

    #### Pose 1 radius 4
    peaks = annotations_1[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=10, radius_head=60 ,mode='Solid')

    ## Reshape a 96x128 pose 1
    image_1 = cv2.resize(image_1, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]
    peaks_resized = []
    #resize dei peaks
    for k in range(len(peaks)):
        p = peaks[k] # coordinate peak ex: "300,200" type string
        c = int(p.split(',')[0])  # column
        r = int(p.split(',')[1])  # row
        if c != -1 and r != -1:
            peaks_resized.append([c / 5 , r / 5]) # 5 è lo scale factor --> 480/96 e 640/128
        else:
            peaks_resized.append([c ,r])
    indices_r4_1, values_r4_1, _ = _getSparsePose(peaks_resized, height, width, config.keypoint_num, radius=2, mode='Solid')  # shape
    pose_mask_r4_1 = cv2.resize(pose_mask_r4_1, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1) #[96, 128, 1]

    # Salvataggio delle 14 heatmap iniziali
    for ii in range(len(peaks)):
        p = peaks[ii]
        if 0 != len(p):
            c = int(p.split(',')[0])  # column
            r = int(p.split(',')[1])  # row
            pose_peaks_1[int(r / h_unit), int(c / w_unit), ii] = 1
            pose_peaks_1_rcv[ii][0] = r
            pose_peaks_1_rcv[ii][1] = c
            pose_peaks_1_rcv[ii][2] = 1


    example = tf.train.Example(features=tf.train.Features(feature={

        'pz_0': dataset_utils.bytes_feature(pz_0.encode('utf-8')), #nome del pz
        'pz_1': dataset_utils.bytes_feature(pz_1.encode('utf-8')),

        'image_name_0': dataset_utils.bytes_feature(name_img_0_16_bit.encode('utf-8')),  # nome dell immagine 0
        'image_name_1': dataset_utils.bytes_feature(name_img_1_16_bit.encode('utf-8')),  # nome dell immagine 1
        'image_raw_0': dataset_utils.bytes_feature(image_0.tostring()),  # immagine 0 in bytes
        'image_raw_1': dataset_utils.bytes_feature(image_1.tostring()),  # immagine 1 in bytes

        'image_format': dataset_utils.bytes_feature('PNG'.encode('utf-8')),
        'image_height': dataset_utils.int64_feature(96),
        'image_width': dataset_utils.int64_feature(128),

        'pose_peaks_0': dataset_utils.float_feature(pose_peaks_0.flatten().tolist()), # immagine con shape [8, 16 14] in cui sono individuati i keypoint dell'immagine 0 NON a raggio 4
        'pose_peaks_1': dataset_utils.float_feature(pose_peaks_1.flatten().tolist()), # immagine con shape [8, 16, 14] in cui sono individuati i keypoint dell'immagine 1 NON a raggio 4

        'pose_mask_r4_0': dataset_utils.int64_feature(pose_mask_r4_0.astype(np.uint16).flatten().tolist()), # maschera binaria a radius 4 con shape [96, 128, 1]
        'pose_mask_r4_1': dataset_utils.int64_feature(pose_mask_r4_1.astype(np.uint16).flatten().tolist()), # maschera binaria a radius 4 con shape [96, 128, 1]

        'indices_r4_0': dataset_utils.bytes_feature(np.array(indices_r4_0).astype(np.int64).tostring()), # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'values_r4_0': dataset_utils.bytes_feature(np.array(values_r4_0).astype(np.int64).tostring()), # coordinate a radius 4 dei keypoints dell'immagine 0, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'indices_r4_1': dataset_utils.bytes_feature(np.array(indices_r4_1).astype(np.int64).tostring()), # coordinate a radius 4 (quindi anche con gli indici del riempimento del keypoint) dei keypoints dell'immagine 1, servono per ricostruire il vettore di sparse [num_indices, 3]
        'values_r4_1': dataset_utils.bytes_feature(np.array(values_r4_1).astype(np.int64).tostring()),
        'shape_len_indices_0': dataset_utils.int64_feature(np.array(indices_r4_0).shape[0]),
        'shape_len_indices_1': dataset_utils.int64_feature(np.array(indices_r4_1).shape[0])

    }))

    return example

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

def fill_tfrecord(lista, tfrecord_writer):

    tot_pairs = 0 # serve per contare il totale di pair nel tfrecord

    # Lettura delle prime annotazioni --> ex:pz3
    for pz_0 in lista:
        name_file_annotation_0 = 'result_pz{id}.csv'.format(id=pz_0)
        path_annotation_0 = os.path.join(config.data_annotations_path, name_file_annotation_0)
        df_annotation_0 = pd.read_csv(path_annotation_0, delimiter=';')

        # Lettura delle seconde annotazioni --> ex:pz4
        for pz_1 in lista:
            if pz_0 != pz_1:
                name_file_annotation_1 = 'result_pz{id}.csv'.format(id=pz_1)
                name_path_annotation_1 = os.path.join(config.data_annotations_path, name_file_annotation_1)
                df_annotation_1 = pd.read_csv(name_path_annotation_1, delimiter=';')

                cnt = 1 # Serve per printare a schermo il numero di example. Lo resettiamo ad uno ad ogni nuovo pz_1

                # Creazione del pair
                for _, row_0 in df_annotation_0.iterrows():

                    # Controllo se l'immagine row_0 contiene i keypoints relativi alla spalla dx e sx e anca dx e sx
                    # In caso di assenza passo all'immagine successiva
                    if check_assenza_keypoints_3_5_10_11(row_0[1:]):
                        continue

                    # lettura random delle row nel secondo dataframe
                    value = randint(0, df_annotation_1.shape[0]-1)
                    row_1 = df_annotation_1.iloc[value]

                    # Controllo se l'immagine row_0 contiene i keypoints relativi alla spalla dx e sx e anca dx e sx
                    # In caso di assenza passo ne seleziona un altra random
                    while check_assenza_keypoints_3_5_10_11(row_1[1:]):
                        value = randint(0, df_annotation_1.shape[0]-1)
                        row_1 = df_annotation_1.iloc[value]

                    sys.stdout.write('\r>> Creazione pair [{pz_0}, {pz_1}] image {cnt}/{tot}'.format(pz_0=pz_0, pz_1=pz_1, cnt=cnt,
                                                                                        tot=df_annotation_0.shape[0]))
                    sys.stdout.flush()
                    # Creazione dell'example tfrecord
                    example = _format_data(config, pz_0, pz_1, row_0, row_1)
                    cnt += 1 # incremento del conteggio degli examples
                    tot_pairs += 1

                    tfrecord_writer.write(example.SerializeToString())

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
    Config_file = __import__('0_config_utils')
    config = Config_file.Config()

    # naming file tfrecord
    output_filename_train = os.path.join(config.data_path, 'BabyPose_train.tfrecord')
    output_filename_valid = os.path.join(config.data_path, 'BabyPose_valid.tfrecord')
    output_filename_test = os.path.join(config.data_path, 'BabyPose_test.tfrecord')

    # liste contenente i num dei pz che vanno considerati per singolo set
    lista_pz_train = [3, 6, 14, 36, 66, 73, 74, 37]
    lista_pz_valid = [4, 7, 29, 30, 43, 76]
    lista_pz_test = [5, 11, 15, 27, 39, 42]

    r_tr = None
    r_v = None
    r_te = None
    if os.path.exists(output_filename_train):
        r_tr = input("Il tf record di train esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_tr == "Y" or r_tr == "N" or r_tr == "y" or r_tr == "n"
    if not os.path.exists(output_filename_train) or r_tr == "Y" or r_tr == "y":
        tfrecord_writer_train = tf.compat.v1.python_io.TFRecordWriter(output_filename_train)
        tot_train = fill_tfrecord(lista_pz_train, tfrecord_writer_train)
        print("TOT TRAIN: ", tot_train)
    elif r_tr == "N" or r_tr == "n":
        print("OK, non farò nulla sul train set")

    if os.path.exists(output_filename_valid):
        r_v = input("Il tf record di valid esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_v == "Y" or r_v == "N" or r_v == "y" or r_v == "n"
    if not os.path.exists(output_filename_valid) or r_v == "Y" or r_v == "y":
        tfrecord_writer_valid = tf.compat.v1.python_io.TFRecordWriter(output_filename_valid)
        tot_valid = fill_tfrecord(lista_pz_valid, tfrecord_writer_valid)
        print("TOT VALID: ", tot_valid)
    elif r_v == "N" or r_v == "n":
        print("OK, non farò nulla sul valid set")

    if os.path.exists(output_filename_test):
        r_te = input("Il tf record di test esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_te == "Y" or r_te == "N" or r_te == "y" or r_te == "n"
    if not os.path.exists(output_filename_test) or r_te == "Y" or r_te == "y":
        tfrecord_writer_test = tf.compat.v1.python_io.TFRecordWriter(output_filename_test)
        tot_test = fill_tfrecord(lista_pz_test, tfrecord_writer_test)
        print("TOT TEST: ", tot_test)
    elif r_te == "N" or r_te == "n":
        print("OK, non farò nulla sul test set")

    dic = {"train": {
        "name_file": config.name_tfrecord_train,
        "list_pz": lista_pz_train,
        "tot": tot_train
    },
        "valid": {
            "name_file": config.name_tfrecord_valid,
            "list_pz": lista_pz_valid,
            "tot": tot_valid
        },

        "test": {
            "name_file": config.name_tfrecord_test,
            "list_pz": lista_pz_test,
            "tot": tot_test
        }
    }
    log_tot_sets = os.path.join(config.data_path, 'pair_tot_sets.pkl')
    f = open(log_tot_sets, "wb")
    pickle.dump(dic, f)
    f.close()




