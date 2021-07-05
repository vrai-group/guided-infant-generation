import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt

from utils import grid
from utils import utils_wgan
from model import G1, G2, Discriminator
from datasets.BabyPose import BabyPose

def predict_G1(config):
    babypose_obj = BabyPose()

    # Preprocess Dataset train
    dataset_train = babypose_obj.get_unprocess_dataset(config.data_path, config.name_tfrecord_train)
    dataset_train = babypose_obj.get_preprocess_predizione(dataset_uno)
    dataset_train = dataset_uno.batch(1)
    dataset_train = dataset_uno.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch

    # Preprocess Dataset train
    dataset = babypose_obj.get_unprocess_dataset(config.data_path, config.name_tfrecord_test)
    dataset = babypose_obj.get_preprocess_predizione(dataset)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model_G1 = G1.build_model(config)
    model_G1.load_weights(os.path.join(config.weigths_path, 'Model_G1_epoch_002-loss_0.263336-maskmse_0.456218-maskssim0.857949-val_loss_0.297772-val_mse_0.369874-val_m_ssim_1.000000.hdf5'))
    cnt = 0

    # Per effettuare incroci tra le img di condizione di test e le pose di train
    # cnt2 = 0
    # p = []  # per raccogliere le pose del train
    # raw1 = []  # per raccogliere le target del train
    # for e in dataset_train:
    #     cnt += 1
    #     X, Y, pz_0, pz_1, _, _, mask_0 = e
    #     pz_0 = pz_0.numpy()[0].decode("utf-8")
    #     pz_1 = pz_1.numpy()[0].decode("utf-8")
    #     print(pz_0, '-', pz_1)
    #
    #     if cnt >= 0:
    #         if pz_0 == "pz3" and pz_1 == "pz11": #salviamo la posa del pz_1
    #
    #             pose_1 = X[:,:,:,1:]
    #             p.append(pose_1)
    #             raw1.append(Y[:,:,:,0])
    #
    #         if len(p) >= 500:
    #             print("Terminata raccolta pose")
    #             for e2 in dataset:
    #
    #                 X_, Y_, pz_0, _,  _, _, mask_0 = e2
    #                 pz_0 = pz_0.numpy()[0].decode("utf-8")
    #
    #                 print(pz_0)
    #
    #                 if pz_0 == "pz37":
    #                     image_raw_0 = X_[:, :, :, 0]
    #                     image_raw_0  = tf.reshape(image_raw_0, (1, 96, 128, 1))
    #
    #                     pose_1 = p[cnt2]
    #                     image_raw_1 = raw1[cnt2]
    #                     cnt2 += 1
    #                     X = tf.concat([image_raw_0, pose_1], axis=-1)
    #                     predizione = model_G1.predict(X, verbose=1)
    #
    #                     # #Unprocess
    #                     image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 400, 32765.5)
    #                     image_raw_0 = tf.cast(image_raw_0, dtype=tf.int16)[0].numpy()
    #
    #                     image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 400, 32765.5)
    #                     image_raw_1 = tf.cast(image_raw_1, dtype=tf.int16)[0].numpy()
    #
    #                     pose_1 = pose_1.numpy()[0]
    #                     pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
    #                     pose_1 = pose_1 / 2
    #                     pose_1 = tf.reshape(pose_1, [96,128,14])*255
    #                     pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    #                     pose_1 = tf.cast(pose_1, dtype=tf.float32)
    #
    #                     predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, 1, 32765.5), clip_value_min=0, clip_value_max=32765)
    #                     predizione = tf.cast(predizione, dtype=tf.float32)[0]
    #
    #                     fig = plt.figure(figsize=(10, 2))
    #                     columns = 4
    #                     rows = 1
    #                     # imgs = [predizione, image_raw_0,  image_raw_1, mask_1]
    #                     # labels = ["Prediction", "Condition image",  "image_raw_1", "mask_1"]
    #                     imgs = [predizione, image_raw_0, image_raw_1, pose_1]
    #                     labels = ["Predizione", "Immagine di condizione", "Target", "Posa desiderata"]
    #                     for i in range(1, columns * rows + 1):
    #                         sub = fig.add_subplot(rows, columns, i)
    #                         sub.set_title(labels[i - 1])
    #                         plt.imshow(imgs[i - 1])
    #                     plt.show()
    #                     #plt.savefig("pred_train/pred_{id}.png".format(id=cnt2,pz_0=pz_0,pz_1=pz_1))

    # Per effettuare le predizioni solamente su dataset di valid/test
    for e in dataset_uno:
        cnt += 1
        X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e
        pz_0 = pz_0.numpy()[0].decode("utf-8")
        pz_1 = pz_1.numpy()[0].decode("utf-8")
        print(pz_0, '-', pz_1)

        if cnt >= 0:
            if pz_0 == "pz3" and pz_1 == "pz11":
                image_raw_0 = X[:, :, :, 0]
                pose_1 = X[:, :, :, 1:]
                image_raw_1 = Y[:, :, :, 0]
                mask_1 = Y[:, :, :, 1]
                predizione = model_G1.predict(X, verbose=1)

                #Unprocess
                image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 400, 32765.5)
                image_raw_0 = tf.cast(image_raw_0, dtype=tf.int16)[0].numpy()

                image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 400, 32765.5)
                image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)[0].numpy()

                pose_1 = pose_1.numpy()[0]
                pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
                pose_1 = pose_1 / 2
                pose_1 = tf.reshape(pose_1, [96,128,14])*255
                pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
                pose_1 = tf.cast(pose_1, dtype=tf.float32)

                mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy()
                mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96,128,1) * 255

                predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, 1, 32765.5), clip_value_min=0, clip_value_max=32765)
                predizione = tf.cast(predizione, dtype=tf.float32)[0]

                fig = plt.figure(figsize=(10, 2))
                columns = 4
                rows = 1
                imgs = [predizione, image_raw_0 + mask_0, pose_1, mask_1]
                labels = ["Predizione", "Immagine di condizione", "Posa desiderata", "Maschera posa desiderata"]
                for i in range(1, columns * rows + 1):
                    sub = fig.add_subplot(rows, columns, i)
                    sub.set_title(labels[i - 1])
                    plt.imshow(imgs[i - 1])
                plt.show()
                #plt.savefig("pred_train/pred_test_epoch_10_{id}.png".format(id=cnt,pz_0=pz_0,pz_1=pz_1))

    ## VIDEO
    # l = None
    # for e in dataset_uno:
    #     cnt += 1
    #     X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e
    #     pz_0 = pz_0.numpy()[0].decode("utf-8")
    #     pz_1 = pz_1.numpy()[0].decode("utf-8")
    #     name_0 = name_0.numpy()[0].decode("utf-8")
    #     name_1 = name_1.numpy()[0].decode("utf-8")
    #     print(pz_0)
    #
    #     if cnt >= 0:
    #         if pz_0 == "pz3" and pz_1 == "pz11":
    #             pose_1 = X[:, :, :, 1:]
    #             p.append((pose_1, name_1))
    #
    #         if len(p) >= 500:
    #             for e2 in dataset:
    #
    #                 X_, Y_, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e2
    #                 pz_0 = pz_0.numpy()[0].decode("utf-8")
    #                 pz_1 = pz_1.numpy()[0].decode("utf-8")
    #                 print(pz_0)
    #
    #                 if pz_0 == "pz42":
    #                     if l == None:
    #                         l = X_[:, :, :, 0]
    #                         p0 = pose_0
    #                     image_raw_0 = l
    #                     image_raw_0 = tf.reshape(image_raw_0, (1, 96, 128, 1))
    #                     pose_0 = p0
    #
    #                     pose_1 = p[cnt2][0]
    #                     name_posa = p[cnt2][1]
    #                     cnt2 += 1
    #                     X = tf.concat([image_raw_0, pose_1], axis=-1)
    #                     predizione = model_G1.predict(X, verbose=1)
    #
    #                     # #Unprocess
    #                     image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 400, 32765.5)
    #                     # image_raw_0 =tf.image.resize(image_raw_0, (480,640))
    #                     image_raw_0 = tf.cast(image_raw_0, dtype=tf.int16)[0].numpy()
    #
    #                     pose_1 = pose_1.numpy()[0]
    #                     pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
    #                     pose_1 = pose_1 / 2
    #                     pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
    #                     pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    #                     pose_1 = tf.cast(pose_1, dtype=tf.float32)
    #
    #                     pose_0 = tf.math.reduce_sum(pose_0, axis=-1).numpy().reshape(96, 128, 1) * 255
    #                     pose_0 = tf.cast(pose_0, dtype=tf.float32)
    #
    #                     mask_1 = tf.reshape(Y_[:, :, :, 1], (96, 128, 1))
    #                     mask_1 = tf.cast(mask_1, dtype=tf.float32)
    #
    #                     predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, 400, 32765.5),
    #                                                   clip_value_min=0, clip_value_max=32765)
    #                     predizione = tf.cast(predizione, dtype=tf.float32)[0]
    #                     # predizione = tf.image.resize(predizione, (480, 640))
    #
    #                     fig = plt.figure(figsize=(80, 30))
    #                     columns = 3
    #                     rows = 1
    #                     imgs = [predizione, image_raw_0 + pose_0, pose_1]
    #                     # labels = ["Prediction", "Condition image",  "image_raw_1", "mask_1"]
    #                     # imgs = [predizione, image_raw_0]
    #                     labels = ["Predizione", "Immagine di condizione", "Posa desiderata"]
    #                     for i in range(1, columns * rows + 1):
    #                         sub = fig.add_subplot(rows, columns, i)
    #                         sub.set_title(labels[i - 1])
    #                         plt.imshow(imgs[i - 1], cmap='gray')
    #                     # plt.show()
    #                     plt.savefig("pred_train/pred_{id}_{posa}.png".format(id=cnt2, posa=name_posa))

                        # fig = plt.figure(figsize=(80, 30))
                        # columns = 1
                        # rows = 1
                        # # imgs = [predizione, image_raw_0,  image_raw_1, mask_1]
                        # # labels = ["Prediction", "Condition image",  "image_raw_1", "mask_1"]
                        # imgs = [predizione]
                        # # labels = ["Predizione", "Immagine di condizione", "Posa desiderata"]
                        # for i in range(1, columns * rows + 1):
                        #     sub = fig.add_subplot(rows, columns, i)
                        #
                        #     # sub.set_title(labels[i - 1])
                        #     plt.setp(sub.get_xticklabels(), visible=False)
                        #     plt.setp(sub.get_yticklabels(), visible=False)
                        #     sub.tick_params(axis='both', which='both', length=0)
                        #     plt.imshow(imgs[i - 1], cmap='gray')
                        # # plt.show()
                        # plt.savefig("pred_one/pred_{id}_{posa}.png".format(id=cnt2, posa=name_posa))
                        #
                        # pos_real = cv2.imread("./data/BabyPose/pz11/"+name_posa.split('16bit')[0]+"8bit.png", 0)
                        # #cv2.imwrite("pred_real/pred_{id}_{posa}".format(id=cnt2, posa=name_posa),pos_real)
                        # fig = plt.figure(figsize=(10, 5))
                        # columns = 1
                        # rows = 1
                        # # imgs = [predizione, image_raw_0,  image_raw_1, mask_1]
                        # # labels = ["Prediction", "Condition image",  "image_raw_1", "mask_1"]
                        # imgs = [pos_real]
                        # # labels = ["Predizione", "Immagine di condizione", "Posa desiderata"]
                        # for i in range(1, columns * rows + 1):
                        #     sub = fig.add_subplot(rows, columns, i)
                        #
                        #     # sub.set_title(labels[i - 1])
                        #     plt.setp(sub.get_xticklabels(), visible=False)
                        #     plt.setp(sub.get_yticklabels(), visible=False)
                        #     sub.tick_params(axis='both', which='both', length=0)
                        #     plt.imshow(imgs[i - 1], cmap='gray')
                        # plt.savefig("pred_real/pred_{id}_{posa}.png".format(id=cnt2, posa=name_posa))

"""
Questo script consente di salvare le predizioni a più ceckpoint (epoche).
Le predizioni sono salvate dallo script nel path: ./pred_<tipo_set>_<giorno_training>/<condition_pz_0>_<target_pz_1>/<epoch>
Le cartelle vengono create in automatico se non presenti.

Lo script preleva i checkpoint nel path: Training/pesi_<giorno_training>/weights
"""
def predict_G1_view_more_epochs(config):
    babypose_obj = BabyPose()

    tipo_set = "train"
    giorno_training = "02_07"
    img_save = 10 #quante img salvare per epoch
    pair = None  # coppie da considerare nelle predizioni, in base al tipo di set vengno settate successivamente
    name_dataset = None # nome del dataset da cosiderare, coppie da considerare nelle predizioni, in base al tipo di set vengno settate successivamente

    if tipo_set == "train":
        pair = ["3-14",  "3-36" , "66-36"]
        name_dataset = config.name_tfrecord_train
    if tipo_set == "valid":
        pair = ["27-42", "39-5", "39-27"]
        name_dataset = config.name_tfrecord_valid
    if tipo_set == "test":
        pair = []
        name_dataset = config.name_tfrecord_test

    if not os.path.exists('pred_' + tipo_set + '_' + giorno_training):
        os.mkdir('pred_' + tipo_set + '_' + giorno_training)

    training_weights_path = os.path.join(config.self.training_weights_path, 'pesi_'+giorno_training, weights)
    dataset = babypose_obj.get_unprocess_dataset(training_weights_path, name_dataset)
    dataset = babypose_obj.get_preprocess_predizione(dataset)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    cnt_img_save = 0
    model_G1 = G1.build_model(config)

    for p in pair:
        p0 = p.split('-')[0]
        p1 = p.split('-')[1]

        for checkpoint in os.listdir(config.weigths_path):
            cnt=0
            epoch = checkpoint.split('epoch_')[1].split('-')[0]
            model_G1.load_weights(os.path.join(config.weigths_path, checkpoint))

            for e in dataset:
                cnt += 1
                X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e
                pz_0 = pz_0.numpy()[0].decode("utf-8")
                pz_1 = pz_1.numpy()[0].decode("utf-8")
                print(pz_0)

                if cnt >= 0:
                        if pz_0 == p0 and pz_1 == p1:
                            image_raw_0 = X[:, :, :, 0]
                            pose_1 = X[:, :, :, 1:]
                            image_raw_1 = Y[:, :, :, 0]
                            mask_1 = Y[:, :, :, 1]
                            predizione = model_G1.predict(X, verbose=1)

                            # Unprocess
                            image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 400, 32765.5)
                            image_raw_0 = tf.cast(image_raw_0, dtype=tf.int16)[0].numpy()

                            image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 400, 32765.5)
                            image_raw_1 = tf.cast(image_raw_1, dtype=tf.int16)[0].numpy()

                            pose_1 = pose_1.numpy()[0]
                            pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
                            pose_1 = pose_1 / 2
                            pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
                            pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
                            pose_1 = tf.cast(pose_1, dtype=tf.float32)

                            mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255
                            mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy()

                            predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, 1, 32765.5), clip_value_min=0, clip_value_max=32765)
                            predizione = tf.cast(predizione, dtype=tf.float32)[0]

                            fig = plt.figure(figsize=(10, 2))
                            columns = 5
                            rows = 1
                            imgs = [predizione, image_raw_0 + mask_0, pose_1, image_raw_1, mask_1]
                            labels = ["Predizione", "Immagine di condizione", "Posa desiderata", "Target", "Maschera posa desiderata"]

                            for i in range(1, columns * rows + 1):
                                sub = fig.add_subplot(rows, columns, i)
                                sub.set_title(labels[i - 1])
                                plt.imshow(imgs[i - 1], cmap='gray')

                            if not os.path.exists('pred_'+tipo_set+'_'+giorno_training+'/'+pz_0+'_'+pz_1):
                                os.mkdir('pred_'+tipo_set+'_'+giorno_training+'/'+pz_0+'_'+pz_1)
                            if not os.path.exists('pred_'+tipo_set+'_'+giorno_training+'/'+pz_0+'_'+pz_1+ '/' + epoch):
                                os.mkdir('pred_'+tipo_set+'_'+giorno_training+'/'+pz_0+'_'+pz_1 + '/' + epoch)
                            plt.savefig('pred_'+tipo_set+'_'+giorno_training+'/'+pz_0+'_'+pz_1+'/'+epoch+"/pred_epoch_{epoch}_{id}_{pz_0}_{pz_1}.png".format(epoch=epoch, id=cnt,pz_0=pz_0,pz_1=pz_1))
                            cnt_img_save += 1
                            if cnt_img_save == img_save:
                                cnt_img_save = 0
                                break



if __name__ == "__main__":
    Config_file = __import__('0_config_utils')
    config = Config_file.Config()

    predict_G1_view_more_epochs(config)
