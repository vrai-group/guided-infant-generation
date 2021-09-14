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
from utils import grid
from Augumentation import apply_augumentation


def predict_G1(config):
    babypose_obj = BabyPose()

    # Preprocess Dataset train
    dataset_train = babypose_obj.get_unprocess_dataset(config.name_tfrecord_train)
    dataset_train = dataset_train.batch(1)
    dataset_train = iter(dataset_train)

    # Preprocess Dataset test
    dataset_valid = babypose_obj.get_unprocess_dataset(config.name_tfrecord_valid)
    dataset_valid = dataset_valid.batch(1)
    dataset_valid = iter(dataset_valid)

    dataset_test = babypose_obj.get_unprocess_dataset(config.name_tfrecord_test)
    dataset_test = dataset_test.batch(1)
    dataset_test = iter(dataset_test)

    #name_tfrecord_aug, dataset_aug_len = apply_augumentation(dataset_test, config, "test")
    #dataset_aug = babypose_obj.get_unprocess_dataset(name_tfrecord_aug)
    dataset_aug_len = 50000
    dataset_aug = babypose_obj.get_unprocess_dataset("BabyPose_train.tfrecord")
    dataset_aug = babypose_obj.get_preprocess_G1_dataset(dataset_aug)
    #dataset_aug = dataset_aug.shuffle(dataset_aug_len // 2, reshuffle_each_iteration=True)
    dataset_aug = dataset_aug.batch(1)
    dataset_aug = iter(dataset_aug)



    model_G1 = G1.build_model()
    #model_G1.load_weights(os.path.join(config.weigths_path, 'Model_G1_epoch_012-loss_0.000486-ssim_0.927942-mask_ssim_0.980482-val_loss_0.000819-val_ssim_0.911594-val_mask_ssim_0.972761.hdf5'))
    model_G1.summary()
    cnt = 0

    #Per effettuare incroci tra le img di condizione di test e le pose di train
    cnt2 = 0
    p = []  # per raccogliere le pose del train
    raw1 = []  # per raccogliere le target del train
    # for e in dataset_valid:
    #     cnt += 1
    #     X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0, mean_0, mean_1 = e
    #     pz_0 = pz_0.numpy()[0].decode("utf-8")
    #     pz_1 = pz_1.numpy()[0].decode("utf-8")
    #     print(pz_0, '-', pz_1)
    #
    #     if cnt >= 0:
    #         if pz_0 == "pz3" and pz_1 == "pz34": #salviamo la posa del pz_1
    #
    #             pose_1 = X[:,:,:,1:]
    #             p.append(pose_1)
    #             raw1.append(Y[:,:,:,0])
    #
    #         if len(p) >= 5:
    #             print("Terminata raccolta pose")
    #             for e2 in dataset_valid:
    #
    #                 X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0, mean_0, mean_1 = e2
    #                 pz_0 = pz_0.numpy()[0].decode("utf-8")
    #
    #                 print(pz_0)
    #
    #                 if pz_0 == "pz111":
    #                     image_raw_0 = X[:, :, :, 0]
    #                     image_raw_0  = tf.reshape(image_raw_0, (1, 96, 128, 1))
    #
    #                     pose_1 = p[cnt2]
    #                     image_raw_1 = raw1[cnt2]
    #                     cnt2 += 1
    #                     X = tf.concat([image_raw_0, pose_1], axis=-1)
    #                     predizione = model_G1.predict(X, verbose=1)
    #
    #                     # #Unprocess
    #                     image_raw_0 = utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5)
    #                     image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint8)[0].numpy()
    #
    #                     image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 400, 32765.5)
    #                     image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint8)[0].numpy()
    #
    #                     pose_1 = pose_1.numpy()[0]
    #                     pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
    #                     pose_1 = pose_1 / 2
    #                     pose_1 = tf.reshape(pose_1, [96,128,14])*255
    #                     pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    #                     pose_1 = tf.cast(pose_1, dtype=tf.float32)
    #
    #                     predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765)
    #                     predizione = tf.cast(predizione, dtype=tf.uint8)[0].numpy()
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
    #                     #cnt += 1
    #                     #cv2.imwrite("a_"+str(cnt)+".png", predizione)
    #                     #plt.savefig("pred_train/pred_{id}.png".format(id=cnt2,pz_0=pz_0,pz_1=pz_1))

    # Per effettuare le predizioni solamente su dataset di valid/test
    for i in range(dataset_aug_len):
        cnt += 1
        batch = next(dataset_aug)
        image_raw_0 = batch[0]  # [batch, 96, 128, 1]
        image_raw_1 = batch[1]  # [batch, 96,128, 1]
        pose_1 = batch[2]  # [batch, 96,128, 14]
        mask_1 = batch[3]  # [batch, 96,128, 1]
        mask_0 = batch[4]  # [batch, 96,128, 1]
        pz_0 = batch[5]  # [batch, 1]
        pz_1 = batch[6]  # [batch, 1]
        name_0 = batch[7]  # [batch, 1]
        name_1 = batch[8]  # [batch, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        pz_0 = pz_0.numpy()[0].decode("utf-8")
        pz_1 = pz_1.numpy()[0].decode("utf-8")
        print(pz_0, '-', pz_1)

        if cnt >= 0:

                input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)
                predizione = model_G1.predict(input_G1, verbose=1)

                # Unprocess
                image_raw_0 = utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5).numpy()
                image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()

                image_raw_1 = tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0,
                                               clip_value_max=32765)
                image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()

                pose_1 = pose_1.numpy()[0]
                pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
                pose_1 = pose_1 / 2
                pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
                pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
                pose_1 = tf.cast(pose_1, dtype=tf.float32)

                mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96, 128, 1)
                mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255

                predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_0, 32765.5), clip_value_min=0,
                                              clip_value_max=32765)
                predizione = tf.cast(predizione, dtype=tf.uint16)[0].numpy()

                fig = plt.figure(figsize=(10, 2))
                columns = 5
                rows = 1
                imgs = [predizione, image_raw_0, pose_1, image_raw_1, mask_1]
                labels = ["Predizione", "Immagine di condizione", "Posa desiderata", "Target", "Maschera posa desiderata"]
                for i in range(1, columns * rows + 1):
                    sub = fig.add_subplot(rows, columns, i)
                    sub.set_title(labels[i - 1])
                    plt.imshow(imgs[i - 1])
                plt.show()
                # plt.savefig("pred_train/pred_test_epoch_10_{id}.png".format(id=cnt,pz_0=pz_0,pz_1=pz_1))


"""
Questo script consente di salvare le predizioni a più ceckpoint (epoche).
Le predizioni sono salvate dallo script nel path: ./pred_<tipo_set>_<giorno_training>/<condition_pz_0>_<target_pz_1>/<epoch>
Le cartelle vengono create in automatico se non presenti.

Lo script preleva i checkpoint nel path: Training/pesi_<giorno_training>/weights
"""
def predict_G1_view_more_epochs(config):
    babypose_obj = BabyPose()

    tipo_set = "valid"
    counter = "13"
    giorno_training = "10_08"
    img_save = 10 #quante img salvare per epoch

    #Path
    pair = None  # coppie da considerare nelle predizioni, in base al tipo di set vengno settate successivamente

    if tipo_set == "train":
        #,"43-73","30-105","30-73","26-17","22-3","17-21", "5-22","5-101", "104-108", "109-110", "30-105","5-5",
        pair = [ "30-73", "43-73", ]
        name_dataset = "BabyPose_train.tfrecord"
    if tipo_set == "valid":
        #"6-7", "27-34","29-34", "24-25",  "66-74","7-14", "20-66",
        #"3-3", "14-14", "27-27", "34-34", "66-66", "110-110"
        #"3-14", "27-34","29-34", "14-110",
        pair = ["14-110",]
        #pair = ["71-14", "20-66", "36-76"]
        name_dataset = "BabyPose_valid.tfrecord"
    if tipo_set == "test":
        pair = []
        name_dataset = "BabyPose_test.tfrecord"

    if not os.path.exists('pred_' + tipo_set + '_' + giorno_training):
        os.mkdir('pred_' + tipo_set + '_' + giorno_training)

    training_weights_path = os.path.join("Training", 'G1','13_Syntetich_esperimenti_mask',counter + '_pesi_'+giorno_training, 'weights')
    dataset = babypose_obj.get_unprocess_dataset(name_dataset)
    dataset = babypose_obj.get_preprocess_predizione_G1(dataset)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model_G1 = G1.build_model()


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
                X, Y, pz_0, pz_1, name_0, name_1, mask_0, pose_0, mean_0, mean_1 = e
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
                            image_raw_0 = utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5)
                            image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()

                            image_raw_1 = utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5)
                            image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()

                            pose_1 = pose_1.numpy()[0]
                            pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
                            pose_1 = pose_1 / 2
                            pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
                            pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
                            pose_1 = tf.cast(pose_1, dtype=tf.float32)

                            mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255
                            mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy()

                            predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765)
                            predizione = tf.cast(predizione, dtype=tf.uint8)[0]

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


def predict_conditional_GAN (config):
    babypose_obj = BabyPose()

    # Preprocess Dataset
    dataset = babypose_obj.get_unprocess_dataset(config.name_tfrecord_train)
    dataset = babypose_obj.get_preprocess_GAN_dataset(dataset)
    dataset = dataset.batch(1)
    dataset = iter(dataset)

    dataset_valid = babypose_obj.get_unprocess_dataset(config.name_tfrecord_valid)
    dataset_valid = babypose_obj.get_preprocess_GAN_dataset(dataset_valid)
    dataset_valid = dataset_valid.batch(1)
    dataset_valid = iter(dataset_valid)

    dataset_test = babypose_obj.get_unprocess_dataset(config.name_tfrecord_test)
    dataset_test = babypose_obj.get_preprocess_GAN_dataset(dataset_test)
    dataset_test = dataset_test.batch(1)
    dataset_test = iter(dataset_test)

    # Carico il modello preaddestrato G1
    model_G1 = G1.build_model()
    model_G1.load_weights(os.path.join(config.weigths_path,'Model_G1_epoch_030-loss_0.000312-ssim_0.788846-mask_ssim_0.982531-val_loss_0.000795-val_ssim_0.730199_val_mask_ssim_0.946310.hdf5'))
    # Carico il modello preaddestrato GAN
    # G2
    model_G2 = G2.build_model()  # architettura Generatore G2
    model_G2.load_weights(os.path.join(config.weigths_path, 'a.hdf5'))
    # D
    model_D = Discriminator.build_model()
    # model_D.load_weights(os.path.join(config.weigths_path, 'b.hdf5'))


    # cnt2 = 0
    # cnt = 0
    # p = []  # per raccogliere le pose del train
    # raw1 = []  # per raccogliere le target del train
    #
    # for id_batch in range(int(config.dataset_train_len / 1)):
    #
    #     batch = next(dataset_valid)
    #     image_raw_0 = batch[0]  # [batch, 96, 128, 1]
    #     image_raw_1 = batch[1]  # [batch, 96,128, 1]
    #     pose_1 = batch[2]  # [batch, 96,128, 14]
    #     mask_1 = batch[3]  # [batch, 96,128, 1]
    #     mask_0 = batch[4]  # [batch, 96,128, 1]
    #     pz_0 = batch[5]  # [batch, 1]
    #     pz_1 = batch[6]  # [batch, 1]
    #     name_0 = batch[7]  # [batch, 1]
    #     name_1 = batch[8]  # [batch, 1]
    #
    #     pz_0 = pz_0.numpy()[0].decode("utf-8")
    #     pz_1 = pz_1.numpy()[0].decode("utf-8")
    #     print(pz_0, '-', pz_1)
    #
    #     if cnt >= 0:
    #         if pz_0 == "pz3" and pz_1 == "pz34": #salviamo la posa del pz_1
    #
    #             p.append(pose_1)
    #
    #         if len(p) >= 5:
    #             print("Terminata raccolta pose")
    #             for id_batch in range(int(config.dataset_train_len / 1)):
    #
    #                 batch = next(dataset_valid)
    #                 image_raw_0 = batch[0]  # [batch, 96, 128, 1]
    #                 image_raw_1 = batch[1]  # [batch, 96,128, 1]
    #                 pose_1 = batch[2]  # [batch, 96,128, 14]
    #                 mask_1 = batch[3]  # [batch, 96,128, 1]
    #                 mask_0 = batch[4]  # [batch, 96,128, 1]
    #                 pz_0 = batch[5]  # [batch, 1]
    #                 pz_1 = batch[6]  # [batch, 1]
    #                 name_0 = batch[7]  # [batch, 1]
    #                 name_1 = batch[8]  # [batch, 1]
    #
    #
    #                 if pz_0 == "pz110":
    #                     # G1
    #                     input_G1 = tf.concat([image_raw_0, p[cnt2]], axis=-1)  # [batch, 96, 128, 15]
    #                     output_G1 = model_G1(input_G1)  # output_g1 --> [batch, 96, 128, 1]
    #                     output_G1 = tf.cast(output_G1, dtype=tf.float16)
    #
    #                     # G2
    #                     input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
    #                     output_G2 = model_G2(input_G2)  # [batch, 96, 128, 1]
    #                     output_G2 = tf.cast(output_G2, dtype=tf.float16)
    #                     refined_result = output_G1 + output_G2
    #
    #                     # Unprocess
    #                     image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 350, 32765.5)
    #                     image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()
    #
    #                     image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 350, 32765.5)
    #                     image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()
    #
    #                     pose_1 = p[cnt2].numpy()[0]
    #                     pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
    #                     pose_1 = pose_1 / 2
    #                     pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
    #                     pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    #                     pose_1 = tf.cast(pose_1, dtype=tf.float32)
    #
    #                     mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96, 128, 1)
    #                     mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255
    #
    #                     refined_result = \
    #                     tf.cast(utils_wgan.unprocess_image(refined_result, 900, 32765.5), dtype=tf.uint16)[0]
    #
    #                     result = tf.image.ssim(refined_result, image_raw_1, max_val=tf.math.reduce_max(refined_result))
    #                     print(result)
    #
    #                     output_G1 = tf.clip_by_value(utils_wgan.unprocess_image(output_G1, 350, 32765.5),
    #                                                  clip_value_min=0,
    #                                                  clip_value_max=32765)
    #                     output_G1 = tf.cast(output_G1, dtype=tf.uint16)[0]
    #
    #                     output_G2 = tf.clip_by_value(utils_wgan.unprocess_image(output_G2, 350, 32765.5),
    #                                                  clip_value_min=0,
    #                                                  clip_value_max=32765)
    #                     output_G2 = tf.cast(output_G2, dtype=tf.uint16)[0]
    #
    #                     # Save img
    #                     # import cv2
    #                     # refined_result = tf.cast((refined_result*32765.5)+350, dtype=tf.uint16)[0]
    #                     # cv2.imwrite("t.png", refined_result.numpy())
    #
    #                     # Predizione D
    #                     # input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
    #                     #                     axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
    #                     # output_D = self.model_D(input_D)  # [batch * 3, 1]
    #                     # output_D = tf.reshape(output_D, [-1])  # [batch*3]
    #                     # output_D = tf.cast(output_D, dtype=tf.float16)
    #                     # D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]
    #
    #                     fig = plt.figure(figsize=(10, 10))
    #                     columns = 6
    #                     rows = 1
    #                     imgs = [output_G1, output_G2, refined_result, pose_1, image_raw_1, image_raw_0]
    #                     for i in range(1, columns * rows + 1):
    #                         fig.add_subplot(rows, columns, i)
    #                         plt.imshow(imgs[i - 1])
    #                     plt.show()
    #                     cnt2+=1


    for id_batch in range(int(config.dataset_train_len / 1)):

        batch = next(dataset)
        if id_batch > 100:
            image_raw_0 = batch[0]  # [batch, 96, 128, 1]
            image_raw_1 = batch[1]  # [batch, 96,128, 1]
            pose_1 = batch[2]  # [batch, 96,128, 14]
            mask_1 = batch[3]  # [batch, 96,128, 1]
            mask_0 = batch[4]  # [batch, 96,128, 1]
            pz_0 = batch[5]  # [batch, 1]
            pz_1 = batch[6]  # [batch, 1]
            name_0 = batch[7]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
            mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))


            print(name_1)
            print(name_0)


            # G1
            input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)  # [batch, 96, 128, 15]
            output_G1 = model_G1(input_G1)  # output_g1 --> [batch, 96, 128, 1]
            output_G1 = tf.cast(output_G1, dtype=tf.float16)

            # G2
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2 = model_G2(input_G2)  # [batch, 96, 128, 1]
            output_G2 = tf.cast(output_G2, dtype=tf.float16)
            refined_result = output_G1 + output_G2

            # Predizione D
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                                axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = model_D(input_D)  # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1])  # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

            print("Reale? ", D_neg_refined_result)


            # Unprocess
            image_raw_0 = utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5)
            image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()

            image_raw_1 = utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5)
            image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()

            pose_1 = pose_1.numpy()[0]
            pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
            pose_1 = pose_1 / 2
            pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
            pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
            pose_1 = tf.cast(pose_1, dtype=tf.float32)

            mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96, 128, 1)
            mask_0 = tf.cast(mask_0, dtype=tf.int16)[0].numpy().reshape(96, 128, 1) * 255

            refined_result = utils_wgan.unprocess_image(refined_result, mean_0, 32765.5)
            refined_result= tf.cast(refined_result, dtype=tf.uint16)[0]

            # result = tf.image.ssim(refined_result, image_raw_1, max_val=tf.math.reduce_max(refined_result))
            # print(result)


            output_G2 = tf.clip_by_value(utils_wgan.unprocess_image(output_G2, mean_0, 32765.5), clip_value_min=0,
                                         clip_value_max=32765)
            output_G2 = tf.cast(output_G2, dtype=tf.uint8)[0]

            output_G1 = tf.clip_by_value(utils_wgan.unprocess_image(output_G1, mean_1, 32765.5), clip_value_min=0,
                                         clip_value_max=32765)
            output_G1 = tf.cast(output_G1, dtype=tf.uint16)[0]



            #Save img
            # import cv2
            # refined_result = tf.cast(refined_result, dtype=tf.uint8)
            # cv2.imwrite("t.png", refined_result.numpy())


            # grid.save_image(tf.reshape(refined_result,(2,96,128)),
            #                 "./t.png")

            fig = plt.figure(figsize=(10, 10))
            columns = 6
            rows = 1
            imgs = [output_G1, output_G2, refined_result, pose_1, image_raw_0, image_raw_1]
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(imgs[i - 1], cmap='gray')
            plt.show()


if __name__ == "__main__":
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    config.print_info()
    #predict_G1_view_more_epochs(config)
    predict_G1(config)
    #predict_conditional_GAN(config)
