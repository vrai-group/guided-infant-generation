import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.morphology import square, dilation, erosion

from utils import format_example, aug_flip, getSparsePose, getSparseKeypoint


def _sparse2dense(indices, values, shape):
    """
    # Data una shape [128, 64, 1] e dati come indices tutti i punti (sia i keypoint che la copertura tra loro),
    # andiamo a creare una maschera binaria in cui solamente la sagoma della persona assuma valore 1
    """
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        dense[r, c, 0] = values[i]
    return dense


def visualizePeaks(peaks, img):
    """
    Serve per stamapare i peaks sull immagine in input. Stampo i peaks considerando anche i corrispettivi id
    """
    for k in range(len(peaks)):
        p = peaks[k]  # coordinate peak ex: "300,200" type string
        x = int(p.split(',')[0])  # column
        y = int(p.split(',')[1])  # row
        if x != -1 and y != -1:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1)
            cv2.imwrite('keypoint.png', img)

    print("Log: Immagine salvata Keypoint")


def _get_segmentation_mask(keypoints, height, width, r_h, r_k, dilatation):
    """
    Consente di creare la maschera di segmentazione (background 0, foreground 255) del bambino effettuando dapprima
    la connessione tra i keypoints e dopodichè applicando operazioni morfologiche di dilatazione ed erosione
    :param keypoints: lista delle 14 annotazioni
    :param int heights:
    :param int width:
    :param int r_h: raggio della testa
    :param int r_k: raggio dei keypoints
    :param int dilatation: valore dell'operazione morfologica di dilatazione
    :return  maschera binaria con shape [height, width, 1]
    """
    # Qui definisco quali keypoints devono essere connessi tra di loro. I numeri fanno riferimento agli ID
    limbSeq = [[0, 3], [0, 4], [0, 5],  # testa
               [1, 2], [2, 3],  # braccio dx
               [3, 4], [4, 5],  # collo
               [5, 6], [6, 7],  # braccio sx
               [11, 12], [12, 13],  # gamba sinistra
               [10, 9], [9, 8],  # gamba destra
               [11, 10],  # Anche
               # Corpo
               [10, 4], [10, 3], [10, 5], [10, 0],
               [11, 4], [11, 3], [11, 5], [11, 0]]
    indices = []
    values = []
    for limb in limbSeq:  # ad esempio limb = [2, 3]
        p0 = keypoints[limb[0]]  # ad esempio coordinate per il keypoint corrispondente con id 2
        p1 = keypoints[limb[1]]  # ad esempio coordinate per il keypoint corrispondente con id 3

        x0 = int(p0.split(',')[0])  # coordinata x  per il punto p0   ex: "280,235"
        y0 = int(p0.split(',')[1])  # coordinata y  per il punto p0
        x1 = int(p1.split(',')[0])  # coordinata x  per il punto p1
        y1 = int(p1.split(',')[1])  # coordinata y  per il punto p1

        # non considero le occlusioni che sono indicate con valore -1
        if y0 != -1 and y1 != -1 and x0 != -1 and x1 != -1:

            if limb[0] == 0:  # Per la testa utilizzo un Radius maggiore
                ind, val = getSparseKeypoint(y0, x0, 0, height, width,
                                             r_h)  # ingrandisco il punto p0 considerando un raggio di 20
            else:
                ind, val = getSparseKeypoint(y1, x1, 0, height, width,
                                             r_k)  # # ingrandisco il punto p1 considerando un raggio di 4

            indices.extend(ind)
            values.extend(val)

            if limb[1] == 0:
                # Per la testa utilizzo un Radius maggiore
                ind, val = getSparseKeypoint(y1, x1, 0, height, width,
                                             r_h)  # ingrandisco il punto p1 considerando un raggio di 20
            else:
                ind, val = getSparseKeypoint(y1, x1, 0, height, width,
                                             r_k)  # ingrandisco il punto p1 considerando un raggio di 4

            indices.extend(ind)
            values.extend(val)

            # Stampo l immagine
            # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
            # cv2.imwrite('Dense{limb}.png'.format(limb=limb), dense * 255)

            # Qui vado a riempire il segmento ad esempio [2,3] corrispondenti ai punti p0 e p1.
            # L idea è quella di riempire questo segmento con altri punti di larghezza r_k.
            # Ovviamente per farlo:
            #   1. calcolo la distanza tra p0 e p1
            #   2. poi vedo quanti punti di raggio r_k entrano in questa distanza
            distance = np.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)  # punto 1.
            sampleN = int(distance / r_k)  # punto 2.
            if sampleN > 1:
                for i in range(1, sampleN):  # per ognuno dei punti di cui ho bisogno
                    y = int(y0 + (y1 - y0) * i / sampleN)  # calcolo della coordinata y
                    x = int(x0 + (x1 - x0) * i / sampleN)  # calcolo della coordinata x
                    # ingrandisco il punto considerando un raggio r_k
                    ind, val = getSparseKeypoint(y, x, 0, height, width, r_k)
                    indices.extend(ind)
                    values.extend(val)
                    # per stampare le connessioni tra i keypoints
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('Linking'+str(limb)+'.png', dense * 255)

            ## stampo l immagine
            dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
            # cv2.imwrite('Dense{limb}.png'.format(limb=limb), dense * 255)

    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
    dense = dilation(dense, square(dilatation))
    dense = erosion(dense, square(5))
    dense = np.reshape(dense, [dense.shape[0], dense.shape[1], 1])
    # cv2.imwrite('DenseMask.png', dense * 255)

    return dense


def _format_data(id_pz_condition, id_pz_target, Ic_annotations, It_annotations, r_k, radius_keypoints_mask,
                 r_h, dilatation):
    """
    Crezione dei dati
    :param int id_pz_condition: id del pz di condizione
    :param int id_pz_target: id del pz target
    :param Ic_annotations: annotazione dei keypoits sull'immagini di condizione
    :param It_annotations: annotazione dei keypoits sull'immagini target
    :param int r_k: raggio della posa
    :param int radius_keypoints_mask: raggio della posa per creare la maschera
    :param int r_h: raggio della testa utilizzato per creare la maschera
    :param int dilation: utilizzato per l'operazione morfologica di dilatazione
    :return dict dic_data: un dizionario in cui sono contenuti i dati creati img_condition, img_target etc..
    """

    pz_condition = 'pz' + str(id_pz_condition)
    pz_target = 'pz' + str(id_pz_target)

    # Read the image info:
    name_img_condition_16_bit = Ic_annotations['image'].split('_')[0] + '_16bit.png'
    name_img_target_16_bit = It_annotations['image'].split('_')[0] + '_16bit.png'
    img_path_condition = os.path.join(dir_dataset, pz_condition, name_img_condition_16_bit)
    img_path_target = os.path.join(dir_dataset, pz_target, name_img_target_16_bit)

    # Read immagine
    Ic = cv2.imread(img_path_condition, cv2.IMREAD_UNCHANGED)  # [480,640]
    It = cv2.imread(img_path_target, cv2.IMREAD_UNCHANGED)  # [480,640]
    height, width = Ic.shape[0], Ic.shape[1]

    # Processamento Image Ic
    keypoints_condition = Ic_annotations[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    Mc = _get_segmentation_mask(keypoints_condition, height, width, r_k=radius_keypoints_mask, r_h=r_h,
                                dilatation=dilatation)  # [480,640,1]

    ## Resize a 96x128
    Ic = cv2.resize(Ic, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]
    keypoints_resized_condition = []
    ## Resize dei peaks
    for i in range(len(keypoints_condition)):
        kp = keypoints_condition[i]  # coordinate peak ex: "300,200" type string
        x = int(kp.split(',')[0])  # column
        y = int(kp.split(',')[1])  # row
        if x != -1 and y != -1:
            keypoints_resized_condition.append([int(x / 5), int(y / 5)])  # 5 è lo scale factor --> 480/96 e 640/128
        else:
            keypoints_resized_condition.append([x, y])
    Ic_indices, Ic_values = getSparsePose(keypoints_resized_condition, height, width, r_k, mode='Solid')
    ## Resize della maschera
    Mc = cv2.resize(Mc, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]

    # Processamento Image It
    keypoints_target = It_annotations[1:]  # annotation_0[1:] --> poichè tolgo il campo image
    Mt = _get_segmentation_mask(keypoints_target, height, width, r_k=radius_keypoints_mask, r_h=r_h,
                                dilatation=dilatation)

    ## Resizea 96x128
    It = cv2.resize(It, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]
    keypoints_resized_target = []
    ## resize dei peaks
    for i in range(len(keypoints_target)):
        kp = keypoints_target[i]  # coordinate peak ex: "300,200" type string
        x = int(kp.split(',')[0])  # column
        y = int(kp.split(',')[1])  # row
        if y != -1 and x != -1:
            keypoints_resized_target.append([int(x / 5), int(y / 5)])  # 5 è lo scale factor --> 480/96 e 640/128
        else:
            keypoints_resized_target.append([x, y])
    It_indices, It_values = getSparsePose(keypoints_resized_target, height, width, r_k, mode='Solid')
    ## Resize mask
    Mt = cv2.resize(Mt, dsize=(128, 96), interpolation=cv2.INTER_NEAREST).reshape(96, 128, 1)  # [96, 128, 1]

    dic_data = {

        'pz_condition': pz_condition,  # nome del pz di condizione
        'pz_target': pz_target,  # nome del pz di target

        'Ic_image_name': name_img_condition_16_bit,  # nome dell immagine di condizione
        'It_image_name': name_img_target_16_bit,  # nome dell immagine di target
        'Ic': Ic,  # immagine di condizione in bytes
        'It': It,  # immagine target in bytes

        'image_format': 'PNG'.encode('utf-8'),
        'image_height': 96,
        'image_width': 128,

        'Ic_original_keypoints': keypoints_resized_condition,
        # valori delle coordinate originali della posa ridimensionati a 96x128
        'It_original_keypoints': keypoints_resized_target,

        'Mc': Mc,  # maschera binaria a radius con shape [96, 128, 1]
        'Mt': Mt,  # maschera binaria a radius 4 con shape [96, 128, 1]

        # Sparse tensors
        # Definizione delle coordinate (quindi anche con gli indici del riempimento del keypoint in base al raggio)
        # dei keypoints dell'immagine di condizione, servono per ricostruire il vettore di sparse, [num_indices, 3]
        'Ic_indices': Ic_indices,
        'Ic_values': Ic_values,
        'It_indices': It_indices,
        'It_values': It_values,

        'radius_keypoints': r_k,  # valore del raggio (r_k) della posa

    }

    return dic_data


def fill_tfrecord(lista, tfrecord_writer, radius_keypoints_pose, radius_keypoints_mask,
                  radius_head_mask, dilatation, campionamento, flip=False, mode="negative"):
    """
    Consente di selezionare la coppia di pair da formare in base alle mode, creare i dati richiamando altri metodi
    e riempire il tfrecord.
    :param list lista: contiene gli id dei pz
    :param tfrecord_writer: writer del tfrecord
    :param int radius_keypoints_pose: raggio della posa
    :param int radius_keypoints_mask: raggio della posa per creare la maschera
    :param int radius_head_mask: raggio della testa utilizzato per creare la maschera
    :param int dilation: utilizzato per l'operazione morfologica di dilatazione
    :param int campionamento: ogni quante immagine devo considerare, utilizzato per diminuire immagini simili
    :param bool flip: se effettuare l'augumentazione di flip
    :param str mode: negative o positive, modalità di accoppiamento pz
    :return int tot_pairs: numero totale di pairs
    """

    tot_pairs = 0  # serve per contare il totale di pair nel tfrecord

    # Accoppiamento tra immagini appartenenti allo stesso pz
    if mode == "positive":
        # TODO da scrivere
        pass

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
                    for indx, row_0 in df_annotation_0.iterrows():

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
                            '\r>Creazione pair [{pz_0}, {pz_1}] image {cnt}/{tot}'.format(pz_0=pz_0, pz_1=pz_1,
                                                                                          cnt=cnt,
                                                                                          tot=df_annotation_0.shape[0]))
                        sys.stdout.flush()

                    print("\n")
            print('\nTerminato {pz_0} \n\n'.format(pz_0=pz_0))

    tfrecord_writer.close()
    print('\nSET DATI TERMINATO\n\n')

    return tot_pairs


if __name__ == '__main__':
    """
    Questo script consente di creare i TFrecord (train/valid/test) che saranno utilizzati per il training e testing
    del modello. Lo script crea un file sets_config.pkl in cui sono contenute tutte le info sul set creato.
    """

    #### CONFIG ##########
    dir_dataset = '../data/Syntetich_complete'
    dir_annotations = '../data/Syntetich_complete/annotations'
    dir_save_tfrecord = '../data/Syntetich_complete/tfrecord/dataset_di_testing'
    keypoint_num = 14

    name_tfrecord_train = 'BabyPose_train.tfrecord'
    name_tfrecord_valid = 'BabyPose_valid.tfrecord'
    name_tfrecord_test = 'BabyPose_test.tfrecord'

    # liste contenente i num dei pz che vanno considerati per singolo set
    lista_pz_train = [101, 103, 105, 106, 107, 109, 110, 112]
    lista_pz_valid = [102, 111]
    lista_pz_test = [104, 108]

    # General information
    r_k = 2  # raggio keypoints
    radius_keypoints_mask = 1
    r_h = 40  # raggio testa nella creazione della maschera
    dilatation = 35
    campionamento = 5
    flip = False  # Aggiunta dell example con flip verticale
    mode = "negative"

    # Check
    assert os.path.exists(dir_dataset)
    assert os.path.exists(dir_annotations)
    if not os.path.exists(dir_save_tfrecord):
        os.mkdir(dir_save_tfrecord)

    #########################

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
        tot_train = fill_tfrecord(lista_pz_train, tfrecord_writer_train, r_k, radius_keypoints_mask,
                                  r_h, dilatation, campionamento, flip=flip, mode=mode)
        print("TOT TRAIN: ", tot_train)
    elif r_tr == "N" or r_tr == "n":
        print("OK, non farò nulla sul train set")

    if os.path.exists(output_filename_valid):
        r_v = input("Il tf record di valid esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_v == "Y" or r_v == "N" or r_v == "y" or r_v == "n"
    if not os.path.exists(output_filename_valid) or r_v == "Y" or r_v == "y":
        tfrecord_writer_valid = tf.compat.v1.python_io.TFRecordWriter(output_filename_valid)
        tot_valid = fill_tfrecord(lista_pz_valid, tfrecord_writer_valid, r_k, radius_keypoints_mask,
                                  r_h, dilatation, campionamento, flip=flip, mode=mode)
        print("TOT VALID: ", tot_valid)
    elif r_v == "N" or r_v == "n":
        print("OK, non farò nulla sul valid set")

    if os.path.exists(output_filename_test):
        r_te = input("Il tf record di test esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_te == "Y" or r_te == "N" or r_te == "y" or r_te == "n"
    if not os.path.exists(output_filename_test) or r_te == "Y" or r_te == "y":
        tfrecord_writer_test = tf.compat.v1.python_io.TFRecordWriter(output_filename_test)
        tot_test = fill_tfrecord(lista_pz_test, tfrecord_writer_test, r_k, radius_keypoints_mask,
                                 r_h, dilatation, campionamento, flip=flip, mode=mode)
        print("TOT TEST: ", tot_test)
    elif r_te == "N" or r_te == "n":
        print("OK, non farò nulla sul test set")

    dic = {

        "general": {
            "radius_keypoints_pose (r_k)": r_k,
            "radius_keypoints_mask": radius_keypoints_mask,
            "radius_head_mask (r_h)": r_h,
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
    log_tot_sets = os.path.join(dir_save_tfrecord, 'sets_config.pkl')
    f = open(log_tot_sets, "wb")
    pickle.dump(dic, f)
    f.close()
