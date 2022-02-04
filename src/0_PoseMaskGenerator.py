"""
Questo script consente di creare i TFrecord (train/valid/test) che saranno utilizzati per il training del modello.
Lo script crea un file pairs.pkl in cui sono contenute tutte le info sui set creati.
Questo file sarà necessario poi per avviare il training.
"""
import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.morphology import square, dilation, erosion

from utils import format_example, aug_flip, getSparsePose, getSparseKeypoint


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


def _getSegmentationMask(peaks, height, width, radius_head=60, radius=4, dilatation=10):
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
                ind, val = getSparseKeypoint(r0, c0, 0, height, width, radius_head)  # ingrandisco il punto p0 considerando un raggio di 20
            else:
                ind, val = getSparseKeypoint(r1, c1, 0, height, width, radius)  # # ingrandisco il punto p1 considerando un raggio di 4

            indices.extend(ind)
            values.extend(val)

            if limb[1] == 0:
                # Per la testa utilizzo un Radius maggiore
                ind, val = getSparseKeypoint(r1, c1, 0, height, width, radius_head)  # ingrandisco il punto p1 considerando un raggio di 20
            else:
                ind, val = getSparseKeypoint(r1, c1, 0, height, width, radius)  # # ingrandisco il punto p1 considerando un raggio di 4

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
                    ind, val = getSparseKeypoint(r, c, 0, height, width, radius)  ## ingrandisco il nuovo punto considerando un raggio di 4
                    indices.extend(ind)
                    values.extend(val)

                    ## per stampare il linking dei lembi
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('Linking'+str(limb)+'.png', dense * 255)
            ## stampo l immagine
            dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
            # cv2.imwrite('Dense{limb}.png'.format(limb=limb), dense * 255)

    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    dense = dilation(dense, square(dilatation))
    dense = erosion(dense, square(5))
    dense = np.reshape(dense, [dense.shape[0], dense.shape[1], 1])
    # cv2.imwrite('DenseMask.png', dense * 255)

    return dense

"""
Crezione dei dati 
@:return dic_data --> un dzionario in cui sono contenuti i dati creati img_0, img_1 etc..
"""


def _format_data(id_pz_0, id_pz_1, Ic_annotations, It_annotations, radius_keypoints_pose, radius_keypoints_mask,
                 radius_head_mask, dilatation):

    pz_condition = 'pz' + str(id_pz_0)
    pz_target = 'pz' + str(id_pz_1)

    # Read the image info:
    name_img_condition_16_bit = Ic_annotations['image'].split('_')[0] + '_16bit.png'
    name_img_target_16_bit = It_annotations['image'].split('_')[0] + '_16bit.png'
    img_path_condition = os.path.join(dir_data, pz_condition, name_img_condition_16_bit)
    img_path_target = os.path.join(dir_data, pz_target, name_img_target_16_bit)

    # Read immagine
    Ic = cv2.imread(img_path_condition, cv2.IMREAD_UNCHANGED)  # [480,640]
    It = cv2.imread(img_path_target, cv2.IMREAD_UNCHANGED)  # [480,640]
    height, width = Ic.shape[0], Ic.shape[1]

    #### Image Ic
    keypoints_condition = Ic_annotations[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    Mc = _getSegmentationMask(keypoints_condition, height, width, radius=radius_keypoints_mask, radius_head=radius_head_mask, dilatation=dilatation)  # [480,640,1]

    #### Resize a 96x128
    Ic = cv2.resize(Ic, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]
    keypoints_resized_condition = []
    # Resize dei peaks a 96x128
    for i in range(len(keypoints_condition)):
        kp = keypoints_condition[i]  # coordinate peak ex: "300,200" type string
        x = int(kp.split(',')[0])  # column
        y = int(kp.split(',')[1])  # row
        if x != -1 and y != -1:
            keypoints_resized_condition.append([int(x / 5), int(y / 5)])  # 5 è lo scale factor --> 480/96 e 640/128
        else:
            keypoints_resized_condition.append([x, y])
    Ic_indices, Ic_values = getSparsePose(keypoints_resized_condition, height, width,  radius_keypoints_pose, mode='Solid')
    # Resize della maschera
    Mc = cv2.resize(Mc, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128,1)  # [96, 128, 1]

    #### Pose image It
    keypoints_target = It_annotations[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    Mt = _getSegmentationMask(keypoints_target, height, width, radius=radius_keypoints_mask, radius_head=radius_head_mask,
                              dilatation=dilatation)

    ## Resizea 96x128
    It = cv2.resize(It, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]
    keypoints_resized_target = []
    # resize dei peaks
    for i in range(len(keypoints_target)):
        kp = keypoints_target[i]  # coordinate peak ex: "300,200" type string
        x = int(kp.split(',')[0])  # column
        y = int(kp.split(',')[1])  # row
        if y != -1 and x != -1:
            keypoints_resized_target.append([int(x / 5), int(y / 5)])  # 5 è lo scale factor --> 480/96 e 640/128
        else:
            keypoints_resized_target.append([x, y])
    It_indices, It_values = getSparsePose(keypoints_resized_target, height, width, radius_keypoints_pose, mode='Solid')
    # Resize mask
    Mt = cv2.resize(Mt, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]

    dic_data = {

        'pz_condition': pz_condition,  # nome del pz di condizione
        'pz_target': pz_target, # nome del pz di target

        'Ic_image_name': name_img_condition_16_bit,  # nome dell immagine di condizione
        'It_image_name': name_img_target_16_bit,  # nome dell immagine di target
        'Ic': Ic,  # immagine di condizione in bytes
        'It': It,  # immagine target in bytes

        'image_format': 'PNG'.encode('utf-8'),
        'image_height': 96,
        'image_width': 128,

        'Ic_original_keypoints': keypoints_resized_condition,  # valori delle coordinate originali della posa ridimensionati a 96x128
        'It_original_keypoints': keypoints_resized_target,

        'Mc': Mc, # maschera binaria a radius con shape [96, 128, 1]
        'Mt': Mt,  # maschera binaria a radius 4 con shape [96, 128, 1]

        # Sparse tensors
        # Definizione delle coordinate (quindi anche con gli indici del riempimento del keypoint in base al raggio)
        # dei keypoints dell'immagine di condizione, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'Ic_indices': Ic_indices,
        'Ic_values': Ic_values,
        'It_indices': It_indices,
        'It_values': It_values,

        'radius_keypoints': radius_keypoints_pose,  # valore del raggio (r_k) della posa

    }

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


"""
Consente di selezionare la coppia di pair da formare
"""


def fill_tfrecord(lista, tfrecord_writer, radius_keypoints_pose, radius_keypoints_mask,
                  radius_head_mask, dilatation, campionamento, flip=False, mode="negative", switch=False):
    tot_pairs = 0  # serve per contare il totale di pair nel tfrecord

    # Accoppiamento tra immagini appartenenti allo stesso pz
    if mode == "positive":
        # TODO da scrivere
        None

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

                    cnt = 0  # Serve per printare a schermo il numero di example. Lo resettiamo ad uno ad ogni nuovo pz_1

                    # Creazione del pair
                    used = []
                    for indx, row_0 in df_annotation_0.iterrows():

                        # row_1=None
                        # if indx < len(df_annotation_1) - 1:
                        #     row_1 = df_annotation_1.loc[indx]
                        # else:
                        #     # lettura random delle row_1 nel secondo dataframe
                        #     conteggio_while = 0
                        #     value = randint(0, len(df_annotation_1) - 1)
                        #     while value in used:
                        #         if conteggio_while < 30:
                        #             value = randint(0, len(df_annotation_1) - 1)
                        #             conteggio_while += 1
                        #         else:
                        #             break
                        #     row_1 = df_annotation_1.loc[value]
                        #     used.append(value)

                        if indx % campionamento == 0:
                            row_1 = df_annotation_1.loc[indx]
                        else:
                            continue

                        # Creazione dell'example tfrecord
                        dic_data = _format_data(pz_0, pz_1, row_0, row_1, radius_keypoints_pose, radius_keypoints_mask,
                                                radius_head_mask, dilatation)
                        example = format_example(dic_data)
                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1  # incremento del conteggio degli examples
                        tot_pairs += 1

                        if flip:
                            dic_data_flip = aug_flip(dic_data.copy())
                            example = format_example(dic_data_flip)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            tot_pairs += 1

                        sys.stdout.write(
                            '\r>> Creazione pair [{pz_0}, {pz_1}] image {cnt}/{tot}'.format(pz_0=pz_0, pz_1=pz_1,
                                                                                            cnt=cnt,
                                                                                            tot=df_annotation_0.shape[
                                                                                                0]))
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
    dir_data = './data/Syntetich_complete'
    dir_annotations = './data/Syntetich_complete/annotations'
    dir_save_tfrecord = './data/Syntetich_complete/tfrecord/negative_no_flip_camp5_key_1_mask_1'
    keypoint_num = 14

    name_tfrecord_train = 'BabyPose_train.tfrecord'
    name_tfrecord_valid = 'BabyPose_valid.tfrecord'
    name_tfrecord_test = 'BabyPose_test.tfrecord'

    # liste contenente i num dei pz che vanno considerati per singolo set
    lista_pz_train = [101, 103, 105, 106, 107, 109, 110, 112]
    lista_pz_valid = [102, 111]
    lista_pz_test = [104, 108]

    # General information
    radius_keypoints_pose = 2 # r_k
    radius_keypoints_mask = 2
    radius_head_mask = 30 # r_h
    dilatation = 25
    campionamento = 5
    flip = False  # Aggiunta dell example con flip verticale
    mode = "negative"
    switch = mode == "positive"  # lo switch è consentito solamente in modalità positive, se è negative va in automatico

    #########################

    # Check
    assert os.path.exists(dir_data)
    assert os.path.exists(dir_annotations)
    if not os.path.exists(dir_save_tfrecord):
        os.mkdir(dir_save_tfrecord)

    # Name of tfrecord file
    output_filename_train = os.path.join(dir_save_tfrecord, name_tfrecord_train)
    output_filename_valid = os.path.join(dir_save_tfrecord, name_tfrecord_valid)
    output_filename_test = os.path.join(dir_save_tfrecord, name_tfrecord_test)

    r_tr, r_v, r_te = None, None, None
    tot_train, tot_valid, tot_test = None, None, None

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
                                  radius_head_mask, dilatation, campionamento, flip=flip, mode=mode, switch=switch)
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
