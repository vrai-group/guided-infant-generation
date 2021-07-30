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
    babypose_obj = BabyPose(config)

    # Preprocess Dataset train
    dataset_train = babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_train)
    dataset_train = babypose_obj.get_preprocess_predizione(dataset_train)
    dataset_train = dataset_train.batch(1)
    dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch

    # Preprocess Dataset test
    dataset = babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_valid)
    dataset = babypose_obj.get_preprocess_predizione(dataset)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model_G1 = G1.build_model(config)
    model_G1.load_weights(os.path.join(config.weigths_path, 'Model_G1_epoch_200-loss_0.000185-mse_inf-ssim_0.094355-mask_ssim_0.811881-val_loss_0.000289-val_mse_inf-val_ssim_0.027771_val_mask_ssim_0.799022.hdf5'))
    #model_G1.load_weights(os.path.join(config.weigths_path,'weights00000650.hdf5'))
    model_G1.summary()
    cnt = 0

    # Per effettuare incroci tra le img di condizione di test e le pose di train
    # cnt2 = 0
    # p = []  # per raccogliere le pose del train
    # raw1 = []  # per raccogliere le target del train
    # for e in dataset_train:
    #     cnt += 1
    #     X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e
    #     pz_0 = pz_0.numpy()[0].decode("utf-8")
    #     pz_1 = pz_1.numpy()[0].decode("utf-8")
    #     print(pz_0, '-', pz_1)
    #
    #     if cnt >= 0:
    #         if pz_0 == "pz109" and pz_1 == "pz108": #salviamo la posa del pz_1
    #
    #             pose_1 = X[:,:,:,1:]
    #             p.append(pose_1)
    #             raw1.append(Y[:,:,:,0])
    #
    #         if len(p) >= 500:
    #             print("Terminata raccolta pose")
    #             for e2 in dataset:
    #
    #                 X_, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e2
    #                 pz_0 = pz_0.numpy()[0].decode("utf-8")
    #
    #                 print(pz_0)
    #
    #                 if pz_0 == "pz66":
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
    for e in dataset:
        cnt += 1
        X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e
        pz_0 = pz_0.numpy()[0].decode("utf-8")
        pz_1 = pz_1.numpy()[0].decode("utf-8")
        print(pz_0, '-', pz_1)

        if cnt >= 0:
            if pz_0 == "pz112" and pz_1 == "pz111":

                if config.input_image_raw_channel == 3:
                    image_raw_0 = X[:, :, :, :3]
                    pose_1 = X[:, :, :, 3:]
                    image_raw_1 = Y[:, :, :, :3]
                    mask_1 = Y[:, :, :, 3]

                elif config.input_image_raw_channel == 1:
                    image_raw_0 = X[:, :, :, 0]
                    pose_1 = X[:, :, :, 1:]
                    image_raw_1 = Y[:, :, :, 0]
                    mask_1 = Y[:, :, :, 1]

                predizione = model_G1.predict(X, verbose=1)

                #Unprocess
                image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 1, 32765.5)
                image_raw_0 = tf.cast(image_raw_0, dtype=tf.float32)[0].numpy()

                image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 1, 32765.5)
                image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)[0].numpy()

                pose_1 = pose_1.numpy()[0]
                pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
                pose_1 = pose_1 / 2
                pose_1 = tf.reshape(pose_1, [96,128,14])*255
                pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
                pose_1 = tf.cast(pose_1, dtype=tf.float32)

                mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96,128,1)
                mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96,128,1) * 255

                predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, 400, 32765.5), clip_value_min=0, clip_value_max=32765)
                predizione = tf.cast(predizione, dtype=tf.uint8)[0]


                fig = plt.figure(figsize=(10, 2))
                columns = 5
                rows = 1
                imgs = [predizione, image_raw_0, pose_1, image_raw_1 , mask_1]
                labels = ["Predizione", "Immagine di condizione", "Posa desiderata", "Target", "Maschera posa desiderata"]
                for i in range(1, columns * rows + 1):
                    sub = fig.add_subplot(rows, columns, i)
                    sub.set_title(labels[i - 1])
                    plt.imshow(imgs[i - 1], cmap='gray')
                plt.show()
                #plt.savefig("pred_train/pred_test_epoch_10_{id}.png".format(id=cnt,pz_0=pz_0,pz_1=pz_1))


"""
Questo script consente di salvare le predizioni a più ceckpoint (epoche).
Le predizioni sono salvate dallo script nel path: ./pred_<tipo_set>_<giorno_training>/<condition_pz_0>_<target_pz_1>/<epoch>
Le cartelle vengono create in automatico se non presenti.

Lo script preleva i checkpoint nel path: Training/pesi_<giorno_training>/weights
"""
def predict_G1_view_more_epochs(config):
    babypose_obj = BabyPose(config)

    tipo_set = "valid"
    counter = "7"
    giorno_training = "23_07"
    dataset = "BabySynt_single_mov_sample"
    type = "positive_single_mov"
    img_save = 10 #quante img salvare per epoch

    #Path
    tfrecord_path = os.path.join('data',dataset,'tfrecord',type)
    pair = None  # coppie da considerare nelle predizioni, in base al tipo di set vengno settate successivamente

    if tipo_set == "train":
        #,"43-73","30-105","30-73","26-17","22-3","17-21", "5-22","5-101","104-108", "109-110", "30-105","5-5",
        pair = ["112-112"]
        name_dataset = "BabyPose_train.tfrecord"
    if tipo_set == "valid":
        #"6-7", "27-34","29-34", "24-25",  "66-74","7-14", "20-66",
        #"3-3", "14-14", "27-27", "34-34", "66-66", "110-110"
        pair = ["110-110"]
        #pair = ["7-14", "20-66", "36-76"]
        name_dataset = "BabyPose_valid.tfrecord"
    if tipo_set == "test":
        pair = []
        name_dataset = "BabyPose_test.tfrecord"

    if not os.path.exists('pred_' + tipo_set + '_' + giorno_training):
        os.mkdir('pred_' + tipo_set + '_' + giorno_training)

    training_weights_path = os.path.join("Training", counter + '_pesi_'+giorno_training, 'weights')
    dataset = babypose_obj.get_unprocess_dataset(tfrecord_path, name_dataset)
    dataset = babypose_obj.get_preprocess_predizione(dataset)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model_G1 = G1.build_model(config)

    for p in pair:
        p0 = p.split('-')[0]
        p1 = p.split('-')[1]

        for checkpoint in os.listdir(training_weights_path):
            cnt=0
            epoch = checkpoint.split('epoch_')[1].split('-')[0]
            model_G1.load_weights(os.path.join(training_weights_path, checkpoint))

            cnt_img_save = 0
            for e in dataset:
                cnt += 1
                X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0 = e
                pz_0 = pz_0.numpy()[0].decode("utf-8")
                pz_1 = pz_1.numpy()[0].decode("utf-8")
                print(pz_0)

                if cnt >= 0:
                        if pz_0 == 'pz'+p0 and pz_1 == 'pz'+p1:
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

                            predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, 400, 32765.5), clip_value_min=0, clip_value_max=32765)
                            predizione = tf.cast(predizione, dtype=tf.float32)[0]

                            fig = plt.figure(figsize=(10, 2))
                            columns = 5
                            rows = 1
                            imgs = [predizione[:,:,0], image_raw_0, pose_1, image_raw_1, mask_1]
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
                                break

    def predict_conditional_GAN(self):

        # Preprocess Dataset test
        dataset_test = self.babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_valid)
        dataset_test = self.babypose_obj.get_preprocess_GAN_dataset(dataset_test)
        dataset_test = dataset_test.batch(1)
        dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch
        test_it = iter(dataset_test)

        # Carico il modello preaddestrato G1
        self.model_G1 = G1.build_model(self.config)
        self.model_G1.load_weights(os.path.join(self.config.weigths_path, 'Model_G1_001-1.029526-5.404243-0.021277-1.276023-6.921112-0.022076.hdf5'))

        # Carico il modello preaddestrato GAN
        self.model_G2 = G2.build_model(self.config)  # architettura Generatore G2
        # self.model_G2.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_0.646448_real_valid_13_real_train_2790.hdf5'))
        self.model_G2.load_weights('./weights/Model_G2_epoch_035-loss_train_2.730875_real_valid_79_real_train_4634.hdf5')
        self.model_D = Discriminator.build_model(self.config)
        # self.model_D.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_2.855738_real_valid_13_real_train_2790.hdf5'))
        self.model_D.load_weights('./weights/Model_D_epoch_035-loss_train_0.659629_real_valid_79_real_train_4634.hdf5')

        for cnt in range(int(self.config.dataset_valid_len / 1)):

            batch = next(test_it)
            image_raw_0 = batch[0]  # [batch, 128,64, 3]
            image_raw_1 = batch[1]  # [batch, 128,64, 3]
            pose_1 = batch[2]  # [batch, 128,64, 18]
            mask_1 = batch[3]  # [batch, 128,64, 1]

            # G1
            input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)  # [batch, 128,64, 21]
            output_G1 = self.model_G1(input_G1)  # output_g1 --> [batch, 128, 64, 3]

            # G2
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 128, 64, 6]
            output_G2 = self.model_G2(input_G2)  # [batch, 128, 64, 3]

            #Deprocess
            output_G1 = tf.clip_by_value((output_G1 * 127.5) + 127, 0, 255)
            output_G1 = tf.cast(output_G1, dtype=tf.int32)[0]

            output_G2 = tf.clip_by_value((output_G2 * 127.5) + 127, 0, 255)
            output_G2 = tf.cast(output_G2, dtype=tf.int32)[0]

            image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 127.5, 127.5)
            image_raw_0  = tf.cast(image_raw_0 , dtype=tf.int32)[0]

            image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 127.5, 127.5)
            image_raw_1 = tf.cast(image_raw_1, dtype=tf.int32)[0]



            refined_result = output_G1 + output_G2

            fig = plt.figure(figsize=(10, 10))
            columns = 5
            rows = 1
            imgs = [output_G1, output_G2, refined_result, image_raw_0, image_raw_1]
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(imgs[i - 1])

            plt.savefig("test.png")
            a = input("Premi per continuare ")



if __name__ == "__main__":
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    #predict_G1_view_more_epochs(config)
    predict_G1(config)
