import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.morphology import square, dilation, erosion

sys.path.append("../src")
from utils import format_example, aug_flip, getSparsePose, enlarge_keypoint


def _sparse2dense(indices, values, shape):
    """
    Create a binary mask in which only the shape of the infant takes on a value of 1
    :param list indices
    :param list values
    :param list shape: shape of dense image
    """
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        dense[r, c, 0] = values[i]
    return dense


def visualizePeaks(keypoints, img):
    """
    Print keypoints on the input image
    :param img: image
    :param series keypoints: keypoints to be printed on the image

    """
    for k in range(len(keypoints)):
        p = keypoints[k]  # coordinate peak ex: "300,200" type string
        x = int(p.split(',')[0])  # column
        y = int(p.split(',')[1])  # row
        if x != -1 and y != -1:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1)
            cv2.imwrite('keypoint.png', img)

    print("Log: Immagine salvata")


def _get_segmentation_mask(keypoints, height, width, r_h, r_k, dilatation):
    """
    It allows the creation of the segmentation mask (background 0, foreground 1) of the child by first
    making the connection between keypoints and then applying morphological operations of dilation and erosion.
    :param series keypoints: list of 14 annotazioni
    :param int heights:
    :param int width:
    :param int r_h: radius of head
    :param int r_k: raggio of keypoints
    :param int dilatation: value of dilatation
    :return  segmentation mask [height, width, 1]
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
                ind, val = enlarge_keypoint(y0, x0, 0, r_h, height,
                                            width)  # ingrandisco il punto p0 considerando un raggio di r_h
            else:
                ind, val = enlarge_keypoint(y1, x1, 0, r_k, height,
                                            width)  # # ingrandisco il punto p1 considerando un raggio di r_k

            indices.extend(ind)
            values.extend(val)

            if limb[1] == 0:
                # Per la testa utilizzo un Radius maggiore
                ind, val = enlarge_keypoint(y1, x1, 0, r_h, height,
                                            width)  # ingrandisco il punto p1 considerando un raggio di 20
            else:
                ind, val = enlarge_keypoint(y1, x1, 0, r_k, height,
                                            width)  # ingrandisco il punto p1 considerando un raggio di 4

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
                    ind, val = enlarge_keypoint(y, x, 0, r_k, height, width)
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
    Creation
    :param int id_pz_condition: id pz of condition image
    :param int id_pz_target: id pz of target image
    :param series Ic_annotations: annotation of keypoints on condition images
    :param series It_annotations: annotation of keypoints on target images
    :param int r_k: radius of pose
    :param int radius_keypoints_mask: radius of pose to create mask
    :param int r_h: head radius used to create the mask
    :param int dilation: used for the morphological dilation operation
    :return dict dic_data: a dictionary in which the data created img_condition, img_target etc. are contained.
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
    It = cv2.imread(img_path_target, cv2.IMREAD_UNCHANGED)
    height, width = Ic.shape[0], Ic.shape[1] # [480,640]

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


def fill_tfrecord(dic_history, lista, tfrecord_writer, radius_keypoints_pose, radius_keypoints_mask,
                  radius_head_mask, dilatation, campionamento, key_dict, flip=False, pairing_mode="negative"):
    """
    Select the pair to be formed by pairing_mode, create the data by calling other methods and fill the tfrecord.
    :param list lista: contiene gli id dei pz
    :param tfrecord_writer: writer del tfrecord
    :param int radius_keypoints_pose: radius of the pose
    :param int radius_keypoints_mask: radius of the pose to crwate the mask
    :param int radius_head_mask: radius of the pose to create the mask head
    :param int dilation: used for the morphological dilation operation
    :param int campionamento: every how many images I have to consider, used to decrease similar images
    :param bool flip: if apply the vertical flip
    :param str pairing_mode: negative or positive, pair mode of pz
    :return int tot_pairs: total number of pairs
    """

    tot_pairs = 0  # serve per contare il totale di pair nel tfrecord

    # Accoppiamento tra immagini appartenenti allo stesso pz
    if pairing_mode == "positive":
        # TODO da scrivere
        pass

    # Accoppiamento tra immagini appartenenti a pz differenti
    if pairing_mode == "negative":
        # Lettura delle prime annotazioni --> ex:pz3
        for id_pz_condition in lista:
            path_annotation_condition = os.path.join(dir_annotations, f'result_pz{id_pz_condition}.csv')
            df_annotation_condition = pd.read_csv(path_annotation_condition, delimiter=';')

            # Lettura delle seconde annotazioni --> ex:pz4
            for id_pz_target in lista:
                if id_pz_condition != id_pz_target:
                    name_path_annotation_target = os.path.join(dir_annotations, f'result_pz{id_pz_target}.csv')
                    df_annotation_target = pd.read_csv(name_path_annotation_target, delimiter=';')

                    cnt = 0  # Serve per printare a schermo il numero di example. Lo resettiamo ad uno ad ogni nuovo pz_target

                    # Creazione del pair
                    for indx, Ic_annotations in df_annotation_condition.iterrows():
                        if indx % campionamento == 0:
                            It_annotations = df_annotation_target.loc[indx]
                        else:
                            continue

                        # Creazione dell'example tfrecord
                        dic_data = _format_data(id_pz_condition, id_pz_target, Ic_annotations, It_annotations,
                                                radius_keypoints_pose, radius_keypoints_mask,
                                                radius_head_mask, dilatation)
                        example = format_example(dic_data)
                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1  # incremento del conteggio degli examples
                        dic_history[f'{key_dict}_{tot_pairs}'] = {'pz_condition': f'pz{id_pz_condition}',
                                                          'img_condition': Ic_annotations['image'],
                                                          'pz_target': f'pz{id_pz_target}',
                                                          'img_target': It_annotations['image'],
                                                          'id_in_tfrecord': f'{key_dict}_{tot_pairs}'}
                        tot_pairs += 1

                        if flip:
                            dic_data_flip = aug_flip(dic_data.copy())
                            example = format_example(dic_data_flip)
                            tfrecord_writer.write(example.SerializeToString())
                            cnt += 1  # incremento del conteggio degli examples
                            dic_history[f'{key_dict}_{tot_pairs}_flipped'] = {'pz_condition': f'pz{id_pz_condition}',
                                                                      'img_condition': Ic_annotations['image'],
                                                                      'pz_target': f'pz{id_pz_target}',
                                                                      'img_target': It_annotations['image'],
                                                                      'id_in_tfrecord': f'{key_dict}_{tot_pairs}'}
                            tot_pairs += 1

                        sys.stdout.write(
                            f'\r>Creazione pair [{id_pz_condition}, {id_pz_target}] image {cnt}/{df_annotation_condition.shape[0] // campionamento}')
                        sys.stdout.flush()

                    print("\n")
            print(f'\nTerminato {id_pz_condition} \n\n')

    tfrecord_writer.close()
    print('\nSET DATI TERMINATO\n\n')

    return tot_pairs


if __name__ == '__main__':
    """
    This script create the dataset configuration. In particular: 
    - TFrecords files (train/valid/test) which will be used for training and testing phase
    - sets_config.pkl file in which all information about the created configuration is contained
    To use the script you must set up the CONFIG part.
    """

    #### CONFIG ##########
    dataset_type = "Syntetich"
    dataset_note = "complete"
    # Specify the dataset configuration name. The name may contain blanks. These will be replaced with the underscore.
    #dataset_configuration = "negative no flip camp 5 keypoints 2 mask 1"
    dataset_configuration = "testing configuration"

    # General information on dataset configuration

    # Specify the [id unique] of the infants you want to have for each set
    lista_pz_train = [101, 103, 105, 106, 107, 109, 110, 112]
    lista_pz_valid = [102, 111]
    lista_pz_test = [104, 108]

    campionamento = 5 # take every 5 images
    r_k = 2  # keypoints radius on Pose maps Pc and Pt
    radius_keypoints_mask = 1
    r_h = 40  # mask radius head
    dilatation = 35  # morphological operation of dilatation
    # if flip == True the script add in tfrecord file the pair flipped the image and related annotation respect vertical axis
    flip = False
    # pairing mode
    # - "neagtive" --> (pz[id unique 1], pz[id unique 2]) [id unique 1] != [id unique 2]
    # - "positive --> (pz[id unique 1], pz[id unique 2])  [id unique 1] == [id unique 2] # NOT IMPLEMENT
    pairing_mode = "negative"

    #########################

    name_dataset = f'{dataset_type}_{dataset_note}'
    dataset_configuration = '_'.join(dataset_configuration.split(" "))
    dir_dataset = os.path.join('.', name_dataset)
    dir_annotations = os.path.join(dir_dataset, 'annotations')
    dir_configuration = os.path.join(dir_dataset, "tfrecord", dataset_configuration)
    keypoint_num = 14

    name_tfrecord_train = f'{dataset_type}_train.tfrecord'
    name_tfrecord_valid = f'{dataset_type}_valid.tfrecord'
    name_tfrecord_test = f'{dataset_type}_test.tfrecord'

    # Check
    assert os.path.exists(dir_dataset)
    assert os.path.exists(dir_annotations)
    for set in [lista_pz_train, lista_pz_valid, lista_pz_test]:
        for id_unique in set:
            assert os.path.exists(os.path.join(dir_dataset, f'pz{id_unique}'))
            assert os.path.exists(os.path.join(dir_annotations, f'result_pz{id_unique}.csv'))
    if not os.path.exists(dir_configuration):
        os.mkdir(dir_configuration)
    assert campionamento != 0

    # Name of tfrecord file
    output_filename_train = os.path.join(dir_configuration, name_tfrecord_train)
    output_filename_valid = os.path.join(dir_configuration, name_tfrecord_valid)
    output_filename_test = os.path.join(dir_configuration, name_tfrecord_test)

    dic_history = {}  # Save, for each set, the positional id of the pair in tferecord file and the name of pz and image paired
    r_tr, r_v, r_te = None, None, None
    tot_train, tot_valid, tot_test = None, None, None

    if os.path.exists(output_filename_train):
        r_tr = input("Il tf record di train esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_tr == "Y" or r_tr == "N" or r_tr == "y" or r_tr == "n"
    if not os.path.exists(output_filename_train) or r_tr == "Y" or r_tr == "y":
        tfrecord_writer_train = tf.compat.v1.python_io.TFRecordWriter(output_filename_train)
        tot_train = fill_tfrecord(dic_history, lista_pz_train, tfrecord_writer_train, r_k, radius_keypoints_mask,
                                  r_h, dilatation, campionamento, key_dict="train", flip=flip,
                                  pairing_mode=pairing_mode)
        print("TOT TRAIN: ", tot_train)
    elif r_tr == "N" or r_tr == "n":
        print("OK, non farò nulla sul train set")

    if os.path.exists(output_filename_valid):
        r_v = input("Il tf record di valid esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_v == "Y" or r_v == "N" or r_v == "y" or r_v == "n"
    if not os.path.exists(output_filename_valid) or r_v == "Y" or r_v == "y":
        tfrecord_writer_valid = tf.compat.v1.python_io.TFRecordWriter(output_filename_valid)
        tot_valid = fill_tfrecord(dic_history, lista_pz_valid, tfrecord_writer_valid, r_k, radius_keypoints_mask,
                                  r_h, dilatation, campionamento, key_dict="valid", flip=flip,
                                  pairing_mode=pairing_mode)
        print("TOT VALID: ", tot_valid)
    elif r_v == "N" or r_v == "n":
        print("OK, non farò nulla sul valid set")

    if os.path.exists(output_filename_test):
        r_te = input("Il tf record di test esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_te == "Y" or r_te == "N" or r_te == "y" or r_te == "n"
    if not os.path.exists(output_filename_test) or r_te == "Y" or r_te == "y":
        tfrecord_writer_test = tf.compat.v1.python_io.TFRecordWriter(output_filename_test)
        tot_test = fill_tfrecord(dic_history, lista_pz_test, tfrecord_writer_test, r_k, radius_keypoints_mask,
                                 r_h, dilatation, campionamento, key_dict="test", flip=flip, pairing_mode=pairing_mode)
        print("TOT TEST: ", tot_test)
    elif r_te == "N" or r_te == "n":
        print("OK, non farò nulla sul test set")

    dic = {

        "general": {
            "campionamento": campionamento,
            "radius_keypoints_pose (r_k)": r_k,
            "radius_keypoints_mask": radius_keypoints_mask,
            "radius_head_mask (r_h)": r_h,
            "dilatation": dilatation,
            "flip": flip,
            "pairing_mode": pairing_mode
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
    set_config_path = os.path.join(dir_configuration, 'sets_config.pkl')
    f = open(set_config_path, "wb")
    pickle.dump(dic, f)
    f.close()
    dic_history_path = os.path.join(dir_configuration, 'dic_history.pkl')
    f = open(dic_history_path, "wb")
    pickle.dump(dic_history, f)
    f.close()
