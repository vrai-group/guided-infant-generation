"""
Questo script consente di avviare il training del G1 e della GAN
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt

from utils import grid
from utils import utils_wgan
from utils import dataset_utils
from model import G1, G2, Discriminator
from datasets.BabyPose import BabyPose
from Augumentation import apply_augumentation


class PG2(object):

    def __init__(self, config):
        self.config = config
        self.babypose_obj = BabyPose()

    def train_G1(self):

        # -Caricamento dataset
        dataset_train = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_train)
        dataset_train = dataset_train.batch(1)

        dataset_valid = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_valid)
        dataset_valid = dataset_valid.batch(1)


        # -Costruzione modello
        self.model_G1 = G1.build_model()
        self.opt_G1 = G1.optimizer()
        self.model_G1.load_weights(os.path.join(self.config.weigths_path, 'Model_G1_epoch_002-loss_0.007832-ssim_0.651670-mask_ssim_0.923238-val_loss_0.003820-val_ssim_0.686539-val_mask_ssim_0.936308.hdf5'))
        #self.model_G1.summary()

        # -History del training
        history_G1 = {'epoch': 0,
                   'loss_train': np.empty((self.config.epochs_G1)),
                   'ssim_train': np.empty((self.config.epochs_G1)),
                   'mask_ssim_train': np.empty((self.config.epochs_G1)),

                   'loss_valid': np.empty((self.config.epochs_G1)),
                   'ssim_valid': np.empty((self.config.epochs_G1)),
                   'mask_ssim_valid': np.empty((self.config.epochs_G1))}

        # Se esistenti, precarico i logs
        if os.path.exists(os.path.join(config.logs_path, 'history_G1.npy')):
            old_history_G1 = np.load(os.path.join(self.config.logs_path,'history_G1.npy'), allow_pickle='TRUE')
            #epoch = old_history_G1[()]['epoch'] --> anche in questo modi riesco ad ottenere il value dell'epoca
            epoch = old_history_G1.item().get('epoch')
            for key, value in old_history_G1.item().items():
                if key == 'epoch':
                    history_G1[key] = value
                else:
                    history_G1[key][:epoch] = value[:epoch]
                    
        


        for epoch in range(history_G1['epoch'],self.config.epochs_G1):
            it_train = iter(dataset_train)
            it_valid = iter(dataset_valid)

            ## Augumentazione Dataset
            name_tfrecord_aug_train, dataset_train_aug_len = apply_augumentation(it_train, config, "train")
            name_tfrecord_aug_valid, dataset_valid_aug_len = apply_augumentation(it_valid, config, "valid")

            print("\nAugumentazione terminata: ")
            print("- lunghezza train: ", dataset_train_aug_len)
            print("- lunghezza valid: ", dataset_valid_aug_len)
            print("\n")

            ## Preprocess Dataset Augumentato
            dataset_train_aug = self.babypose_obj.get_unprocess_dataset(name_tfrecord_aug_train)
            dataset_train_aug = dataset_train_aug.shuffle(dataset_train_aug_len//2, reshuffle_each_iteration=True)
            dataset_train_aug = self.babypose_obj.get_preprocess_G1_dataset(dataset_train_aug)
            dataset_train_aug = dataset_train_aug.batch(self.config.batch_size_train)
            dataset_train_aug = dataset_train_aug.prefetch(tf.data.AUTOTUNE)

            dataset_valid_aug = self.babypose_obj.get_unprocess_dataset(name_tfrecord_aug_valid)
            dataset_valid_aug = self.babypose_obj.get_preprocess_G1_dataset(dataset_valid_aug)
            dataset_valid_aug = dataset_valid_aug.batch(self.config.batch_size_valid)
            dataset_valid_aug = dataset_valid_aug.prefetch(tf.data.AUTOTUNE)

            # numero di batches nel dataset
            num_batches_train = dataset_train_aug_len // self.config.batch_size_train
            num_batches_valid = dataset_valid_aug_len // self.config.batch_size_valid

            train_it = iter(dataset_train_aug)  # rinizializzo l iteratore sul train dataset
            valid_it = iter(dataset_valid_aug)  # rinizializzo l iteratore sul valid dataset


            # Vettori che mi serviranno per salvare i valori per ogni epoca in modo tale da printare a schermo le medie
            logs_to_print = {'loss_values_train': np.empty((num_batches_train)),
                             'ssim_train': np.empty((num_batches_train)),
                             'mask_ssim_train': np.empty((num_batches_train)),

                             'loss_values_valid': np.empty((num_batches_valid)),
                             'ssim_valid': np.empty((num_batches_valid)),
                             'mask_ssim_valid': np.empty((num_batches_valid))
                             }

            # Train
            for id_batch in range(num_batches_train):
                logs_to_print['loss_values_train'][id_batch], logs_to_print['ssim_train'][id_batch],\
                logs_to_print['mask_ssim_train'][id_batch] = self._train_step_G1(train_it, epoch, id_batch)

                # Logs a schermo
                sys.stdout.write('\r')
                sys.stdout.write('Epoch {epoch} step {id_batch} / {num_batches} --> loss_G1: {loss_G1:2f}, '
                                 'ssmi: {ssmi:2f}, mask_ssmi: {mask_ssmi:2f}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch+1,
                    num_batches=num_batches_train,
                    loss_G1=np.mean(logs_to_print['loss_values_train'][:id_batch + 1]),
                    ssmi=np.mean(logs_to_print['ssim_train'][:id_batch + 1]),
                    mask_ssmi=np.mean(logs_to_print['mask_ssim_train'][:id_batch + 1])))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.write('Validazione..')
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                logs_to_print['loss_values_valid'][id_batch], \
                logs_to_print['ssim_valid'][id_batch], logs_to_print['mask_ssim_valid'][id_batch] \
                    = self._valid_step_G1(valid_it, epoch, id_batch)

                sys.stdout.write('\r')
                sys.stdout.write('{id_batch} / {total}'.format(id_batch=id_batch+1, total=num_batches_valid))
                sys.stdout.flush()

            sys.stdout.write('\r\r')
            sys.stdout.write('val_loss_G1: {loss_G1:2f}, val_ssmi: {ssmi:2f}, val_mask_ssmi: {mask_ssmi:2f}'.format(
                loss_G1=np.mean(logs_to_print['loss_values_valid']),
                ssmi=np.mean(logs_to_print['ssim_valid']),
                mask_ssmi=np.mean(logs_to_print['mask_ssim_valid'])))
            sys.stdout.flush()
            sys.stdout.write('\n\n')

            # -CallBacks
            # --Save weights
            name_model = 'Model_G1_epoch_{epoch:03d}-' \
                         'loss_{loss:2f}-' \
                         'ssim_{m_ssim:2f}-' \
                         'mask_ssim_{mask_ssim:2f}-' \
                         'val_loss_{val_loss:2f}-' \
                         'val_ssim_{val_m_ssim:2f}-' \
                         'val_mask_ssim_{val_mask_ssim:2f}.hdf5'.format(
                    epoch=epoch + 1,
                    loss=np.mean(logs_to_print['loss_values_train']),
                    m_ssim=np.mean(logs_to_print['ssim_train']),
                    mask_ssim=np.mean(logs_to_print['mask_ssim_train']),
                    val_loss=np.mean(logs_to_print['loss_values_valid']),
                    val_m_ssim=np.mean(logs_to_print['ssim_valid']),
                    val_mask_ssim=np.mean(logs_to_print['mask_ssim_valid']))
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_G1.save_weights(filepath)

            # --Update learning rate
            if epoch % self.config.lr_update_epoch_G1 == self.config.lr_update_epoch_G1 - 1:
                self.opt_G1.lr = self.opt_G1.lr * self.config.drop_rate_G1
                print("-Aggiornamento Learning rate G1: ", self.opt_G1.lr.numpy())
                print("\n")

            # --Save logs
            history_G1['epoch'] = epoch + 1
            history_G1['loss_train'][epoch] = np.mean(logs_to_print['loss_values_train'])
            history_G1['ssim_train'][epoch] = np.mean(logs_to_print['ssim_train'])
            history_G1['mask_ssim_train'][epoch] = np.mean(logs_to_print['mask_ssim_train'])
            history_G1['loss_valid'][epoch] = np.mean(logs_to_print['loss_values_valid'])
            history_G1['ssim_valid'][epoch] = np.mean(logs_to_print['ssim_valid'])
            history_G1['mask_ssim_valid'][epoch] = np.mean(logs_to_print['mask_ssim_valid'])
            np.save(os.path.join(self.config.logs_path,'history_G1.npy'), history_G1)

            # --Save Gooogle colab
            if self.config.run_google_colab and (
                    epoch % self.config.download_weight == self.config.download_weight - 1):
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar ./logs -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar ./results_ssim -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar weights/ -idq')
                print("-RAR creato\n")
                #Pulizia enviroment
                os.system('rm -r weights/*.hdf5')
                os.system('rm -r results_ssim/G1/train results_ssim/G1/valid')
                os.system('rm -r results_ssim/G1/train results_ssim/G1/valid')
                os.system('mkdir results_ssim/G1/train results_ssim/G1/valid')
                print("-Pulizia enviroment eseguita\n")

        print("#############\n\n")



    def _train_step_G1(self, train_it, epoch, id_batch):

        batch = next(train_it)
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

        with tf.GradientTape() as g1_tape:

            # G1
            input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)
            output_G1 = self.model_G1(input_G1)  # [batch, 96, 128, 1] dtype=float32

            # Loss G1
            loss_value_G1 = G1.PoseMaskLoss1(output_G1, image_raw_1, mask_1)


        self.opt_G1.minimize(loss_value_G1, var_list=self.model_G1.trainable_weights, tape=g1_tape)

        # Metrics
        # - SSIM
        ssim_value = G1.m_ssim(output_G1, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G1.mask_ssim(output_G1, image_raw_1, mask_1, mean_0, mean_1)

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_train == self.config.save_grid_ssim_epoch_train - 1:
            name_directory = os.path.join("./results_ssim/G1/train", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G1_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         ssim=ssim_value,
                                         mask_ssim=mask_ssim_value))
            mean_0 = tf.cast(mean_0, dtype=tf.float32)
            output_G1 = utils_wgan.unprocess_image(output_G1, mean_0, 32765.5)
            grid.save_image(output_G1,
                            name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

            stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
            stack_pairs = np.array(
                [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
                 stack_pairs])
            txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
            file = open(name_directory + '/' + 'G1_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                           batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G1, ssim_value, mask_ssim_value

    def _valid_step_G1(self, valid_it, epoch, id_batch):

        batch = next(valid_it)
        image_raw_0 = batch[0]  # [batch, 96,128, 1]
        image_raw_1 = batch[1]  # [batch, 96,128, 1]
        pose_1 = batch[2]  # [batch, 96,128, 1]
        mask_1 = batch[3]  # [batch, 96,128, 1]
        mask_0 = batch[4]  # [batch, 96,128, 1]
        pz_0 = batch[5]  # [batch, 1]
        pz_1 = batch[6]  # [batch, 1]
        name_0 = batch[7]  # [batch, 1]
        name_1 = batch[8]  # [batch, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)
        output_G1 = self.model_G1(input_G1)  # [batch, 96, 128, 1] dtype=float32

        # Loss G1
        loss_value_G1 = G1.PoseMaskLoss1(output_G1, image_raw_1, mask_1)

        # Metrics
        # - SSIM
        ssim_value = G1.m_ssim(output_G1, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G1.mask_ssim(output_G1, image_raw_1, mask_1, mean_0, mean_1)

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_valid == self.config.save_grid_ssim_epoch_valid - 1:
            name_directory = os.path.join("./results_ssim/G1/valid", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G1_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         ssim=ssim_value,
                                         mask_ssim=mask_ssim_value))
            mean_0 = tf.cast(mean_0, dtype=tf.float32)
            output_G1 = utils_wgan.unprocess_image(output_G1, mean_0, 32765.5)
            grid.save_image(output_G1,
                            name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

            stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
            stack_pairs = np.array(
                [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
                 stack_pairs])
            txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
            file = open(name_directory + '/' + 'G1_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                           batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G1, ssim_value, mask_ssim_value

    def train_conditional_GAN(self):

        # Note: G1 è preaddestrato

        # Preprocess Dataset train
        dataset_train = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_train)
        dataset_train = dataset_train.shuffle(self.config.dataset_train_len, reshuffle_each_iteration=True)
        dataset_train = self.babypose_obj.get_preprocess_GAN_dataset(dataset_train)
        dataset_train = dataset_train.batch(self.config.batch_size_train)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

        # Preprocess Dataset valid
        dataset_valid = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_valid)
        dataset_valid = self.babypose_obj.get_preprocess_GAN_dataset(dataset_valid)
        dataset_valid = dataset_valid.batch(self.config.batch_size_valid)
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)

        # Carico il modello preaddestrato G1
        self.model_G1 = G1.build_model()
        self.model_G1.load_weights(os.path.join(self.config.weigths_path,
                                                 'Model_G1_epoch_010-loss_0.001224-ssim_0.595070-mask_ssim_0.941551-val_loss_0.001531-val_ssim_0.592082_val_mask_ssim_0.929145.hdf5'))

        # Buildo la GAN
        # G2
        self.model_G2 = G2.build_model()  # architettura Generatore G2
        #self.model_G2.summary()
        # self.model_G2.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_0.646448_real_valid_13_real_train_2790.hdf5'))
        self.opt_G2 = G2.optimizer()  # ottimizzatore

        # D
        self.model_D = Discriminator.build_model()
        # self.model_D.summary()
        #self.model_D.load_weights(os.path.join(self.config.weigths_path, 'Model_D_epoch_005-loss_0.483811-loss_values_D_fake_0.275987-loss_values_D_real_0.207833-val_loss_0.497179-val_loss_values_D_fake_0.176609-val_loss_values_D_real_0.320597.hdf5'))
        self.opt_D = Discriminator.optimizer()

        num_batches_train = int(
            self.config.dataset_train_len / self.config.batch_size_train)  # numero di batches nel dataset di train
        num_batches_valid = int(
            self.config.dataset_valid_len / self.config.batch_size_valid)  # numero di batches nel dataset di valid

        # Logs da salvare nella cartella logs per ogni epoca
        logs_loss_train_G2 = np.empty((self.config.epochs_GAN))
        logs_loss_train_D = np.empty((self.config.epochs_GAN))
        logs_loss_train_D_fake = np.empty((self.config.epochs_GAN))
        logs_loss_train_D_real = np.empty((self.config.epochs_GAN))
        logs_mask_ssim = np.empty((self.config.epochs_GAN))
        logs_ssim = np.empty((self.config.epochs_GAN))
        logs_r_r = np.empty((self.config.epochs_GAN))
        logs_img_0 = np.empty((self.config.epochs_GAN))
        logs_img_1 = np.empty((self.config.epochs_GAN))
        if os.path.exists(os.path.join(config.logs_path, 'logs_loss_train_G2.npy')):
            # Se esistenti, precarico i logs
            a = np.load(os.path.join(config.logs_path, 'logs_loss_train_G2.npy'))
            num = a.shape[0]
            logs_loss_train_G2[:num] = np.load(os.path.join(config.logs_path, 'logs_loss_train_G2.npy'))
            logs_loss_train_D[:num] = np.load(os.path.join(config.logs_path, 'logs_loss_train_D_total.npy'))
            logs_loss_train_D_fake[:num] = np.load(os.path.join(config.logs_path, 'logs_loss_train_D_fake.npy'))
            logs_loss_train_D_real[:num] = np.load(os.path.join(config.logs_path, 'logs_loss_train_D_fake.npy'))
            logs_mask_ssim[:num] = np.load(os.path.join(config.logs_path, 'logs_mask_ssim.npy'))
            logs_ssim[:num] = np.load(os.path.join(config.logs_path, 'logs_ssim.npy'))
            logs_r_r[:num] = np.load(os.path.join(config.logs_path, 'logs_r_r.npy'))
            logs_img_0[:num] = np.load(os.path.join(config.logs_path, 'logs_img_0.npy'))
            logs_img_1[:num] = np.load(os.path.join(config.logs_path, 'logs_img_1.npy'))


        for epoch in range(self.config.epochs_GAN):
            train_it = iter(dataset_train)  # rinizializzo l iteratore sul train dataset
            valid_it = iter(dataset_valid)  # rinizializzo l iteratore sul valid dataset

            mean_loss_G2_train = 0  # calcolo la media della loss ad ogni iterazione sul batch
            mean_loss_D_train = 0  # calcolo la media  della loss ad ogni iterazione sul batch
            mean_loss_D_train_fake = 0
            mean_loss_D_train_real = 0

            mean_ssim_train = 0
            mean_mask_ssim_train = 0
            cnt_predette_refined_result_train = 0  # counter per le reali (output_G1 + output_G2) predette nel train
            cnt_predette_image_raw_0_train = 0
            cnt_predette_image_raw_1_train = 0

            mean_ssim_valid = 0
            mean_mask_ssim_valid = 0
            cnt_predette_refined_result_valid = 0  # counter per le reali (output_G1 + output_G2) predette nel train
            cnt_predette_image_raw_0_valid = 0
            cnt_predette_image_raw_1_valid = 0

            # Mi servono per il calcolo della media della loss per ogni batch da printare a schermo
            loss_values_train_G2 = np.empty((num_batches_train))
            loss_values_train_D = np.empty((num_batches_train))
            loss_values_train_D_fake = np.empty((num_batches_train))
            loss_values_train_D_real = np.empty((num_batches_train))
            ssim_train = np.empty((num_batches_train))
            mask_ssim_train = np.empty((num_batches_train))

            loss_values_valid_G2 = np.empty((num_batches_valid))
            loss_values_valid_D = np.empty((num_batches_valid))
            loss_values_valid_D_fake = np.empty((num_batches_valid))
            loss_values_valid_D_real = np.empty((num_batches_valid))
            ssim_valid = np.empty((num_batches_valid))
            mask_ssim_valid = np.empty((num_batches_valid))

            # if self.cnt_d == 2:
            #     self.cnt_d = 0
            # else:
            #     self.cnt_d += 1

            # Train
            for id_batch in range(num_batches_train):
                loss_values_train_G2[id_batch], loss_values_train_D[id_batch], \
                loss_values_train_D_fake[id_batch], loss_values_train_D_real[id_batch], \
                real_predette_refined_result_train, real_predette_image_raw_0_train, real_predette_image_raw_1_train, \
                ssim_train[id_batch], mask_ssim_train[id_batch] = self._train_step(train_it, epoch, id_batch)


                # Calcolo media
                # Loss
                mean_loss_G2_train = np.mean(loss_values_train_G2[:id_batch + 1])
                mean_loss_D_train = np.mean(loss_values_train_D[:id_batch + 1])
                mean_loss_D_train_fake = np.mean(loss_values_train_D_fake[:id_batch + 1])
                mean_loss_D_train_real = np.mean(loss_values_train_D_real[:id_batch + 1])

                # Metrics
                cnt_predette_refined_result_train += real_predette_refined_result_train
                cnt_predette_image_raw_0_train += real_predette_image_raw_0_train
                cnt_predette_image_raw_1_train += real_predette_image_raw_1_train
                mean_ssim_train = np.mean(ssim_train[:id_batch + 1])
                mean_mask_ssim_train = np.mean(mask_ssim_train[:id_batch + 1])

                # Logs a schermo
                sys.stdout.write('\r')
                sys.stdout.write('Epoch {epoch} step {id_batch} / {num_batches} --> loss_G2: {loss_G2:2f}, '
                                 'loss_D: {loss_D:2f}, loss_D_fake: {loss_D_fake:2f}, loss_D_real: {loss_D_real:2f},'
                                 'ssmi: {ssmi:2f}, mask_ssmi: {mask_ssmi:2f},'
                                 'real_predette:: r_r:{r_r:1}, im_0:{im_0:1}, im_1:{im_1:1} / {total_train}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch, num_batches=num_batches_train, loss_G2=mean_loss_G2_train,
                    loss_D=mean_loss_D_train, loss_D_fake=mean_loss_D_train_fake, loss_D_real=mean_loss_D_train_real,
                    r_r=cnt_predette_refined_result_train, im_0=cnt_predette_image_raw_0_train,
                    im_1=cnt_predette_image_raw_1_train, ssmi=mean_ssim_train, mask_ssmi=mean_mask_ssim_train,
                    total_train=self.config.dataset_train_len))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.write('Validazione..')
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                loss_values_valid_G2[id_batch], loss_values_valid_D[id_batch], \
                loss_values_valid_D_fake[id_batch], loss_values_valid_D_real[id_batch], \
                real_predette_refined_result_valid, real_predette_image_raw_0_valid, real_predette_image_raw_1_valid, \
                ssim_valid[id_batch], mask_ssim_valid[id_batch] = self._valid_step(valid_it, epoch, id_batch)

                # Metrics
                cnt_predette_refined_result_valid += real_predette_refined_result_valid
                cnt_predette_image_raw_0_valid += real_predette_image_raw_0_valid
                cnt_predette_image_raw_1_valid += real_predette_image_raw_1_valid

                sys.stdout.write('\r')
                sys.stdout.write('{id_batch} / {total}'.format(id_batch=id_batch, total=num_batches_valid))
                sys.stdout.flush()

            # Calcolo media
            # Loss
            mean_loss_G2_valid = np.mean(loss_values_valid_G2)
            mean_loss_D_valid = np.mean(loss_values_valid_D)
            mean_loss_D_valid_fake = np.mean(loss_values_valid_D_fake)
            mean_loss_D_valid_real = np.mean(loss_values_valid_D_real)

            # Metrics
            mean_ssim_valid = np.mean(ssim_valid)
            mean_mask_ssim_valid = np.mean(mask_ssim_valid)

            sys.stdout.write('\r')
            sys.stdout.write('\r')
            sys.stdout.write('val_loss_G2: {loss_G2:2f}, val_loss_D: {loss_D:2f}, val_loss_D_fake: {loss_D_fake:2f}, '
                             'val_loss_D_real: {loss_D_real:2f}, val_ssmi: {ssmi:2f}, val_mask_ssmi: {mask_ssmi:2f} \n\n'
                             'val_real_predette: r_r:{r_r:1}, im_0:{im_0:1}, im_1:{im_1:1} / {total_valid}'.format(
                loss_G2=mean_loss_G2_valid, loss_D=mean_loss_D_valid,
                loss_D_fake=mean_loss_D_valid_fake, loss_D_real=mean_loss_D_valid_real,
                r_r=cnt_predette_refined_result_valid, im_0=cnt_predette_image_raw_0_valid,
                im_1=cnt_predette_image_raw_1_valid, ssmi=mean_ssim_valid, mask_ssmi=mean_mask_ssim_valid,
                total_valid=self.config.dataset_valid_len))
            sys.stdout.flush()
            sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.write('\n')

            # Save weights
            name_model = "Model_G2_epoch_{epoch:03d}-" \
                         "loss_{loss:2f}-ssmi_{ssmi:2f}-" \
                         "mask_ssmi_{mask_ssim:2f}-" \
                         "r_r_train_{r_r}-" \
                         "im_0_train_{im_0}-" \
                         "im_1_train_{im_1}-" \
                         "val_loss_{val_loss:2f}-" \
                         "val_ssim_{val_ssim:2f}-" \
                         "val_mask_ssim_{val_mask_ssim:2f}-" \
                         "val_r_r_train_{val_r_r}-" \
                         "val_im_0_train_{val_im_0}-" \
                         "val_im_1_train_{val_im_1}.hdf5".format(
                epoch=epoch + 1, loss=mean_loss_G2_train, ssmi=mean_ssim_train, mask_ssim=mean_mask_ssim_train,
                r_r=cnt_predette_refined_result_train, im_0=cnt_predette_image_raw_0_train,
                im_1=cnt_predette_image_raw_1_train,
                val_loss=mean_loss_G2_valid, val_ssim=mean_ssim_valid, val_mask_ssim=mean_mask_ssim_valid,
                val_r_r=cnt_predette_refined_result_valid, val_im_0=cnt_predette_image_raw_0_valid,
                val_im_1=cnt_predette_image_raw_1_valid,
            )
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_G2.save_weights(filepath)

            # loss_values_valid_D[id_batch], \
            # loss_values_valid_D_fake[id_batch], loss_values_valid_D_real
            name_model = "Model_D_epoch_{epoch:03d}-" \
                         "loss_{loss:2f}-" \
                         "loss_values_D_fake_{loss_D_fake:2f}-" \
                         "loss_values_D_real_{loss_D_real:2f}-" \
                         "val_loss_{val_loss:2f}-" \
                         "val_loss_values_D_fake_{val_loss_D_real:2f}-" \
                         "val_loss_values_D_real_{val_loss_D_fake:2f}.hdf5".format(
                epoch=epoch + 1, loss=mean_loss_D_train, loss_D_fake=mean_loss_D_train_fake,
                loss_D_real=mean_loss_D_train_real,
                val_loss=mean_loss_D_valid, val_loss_D_real=mean_loss_D_valid_real,
                val_loss_D_fake=mean_loss_D_valid_fake)
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_D.save_weights(filepath)

            # Save logs
            logs_loss_train_G2[epoch] = mean_loss_G2_train
            logs_loss_train_D[epoch] = mean_loss_D_train
            logs_loss_train_D_fake[epoch] = mean_loss_D_train_fake
            logs_loss_train_D_real[epoch] = mean_loss_D_train_real
            logs_mask_ssim[epoch] = mean_ssim_train
            logs_ssim[epoch] = mean_ssim_train
            logs_r_r[epoch] = cnt_predette_refined_result_train
            logs_img_0[epoch] = cnt_predette_image_raw_0_train
            logs_img_1[epoch] = cnt_predette_image_raw_1_train

            np.save(os.path.join(self.config.logs_path, 'logs_loss_train_G2.npy'), logs_loss_train_G2[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_loss_train_D_total.npy'), logs_loss_train_D[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_loss_train_D_fake.npy'),
                    logs_loss_train_D_fake[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_loss_train_D_real.npy'),
                    logs_loss_train_D_real[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_mask_ssim.npy'), logs_mask_ssim[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_ssim.npy'), logs_ssim[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_r_r.npy'), logs_r_r[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_img_0.npy'), logs_img_0[:epoch + 1])
            np.save(os.path.join(self.config.logs_path, 'logs_img_1.npy'), logs_img_1[:epoch + 1])

            # Update learning rate
            if epoch % self.config.lr_update_epoch_GAN == self.config.lr_update_epoch_GAN - 1:
                self.opt_G2.lr = self.opt_G2.lr * 0.5
                self.opt_D.lr = self.opt_D.lr * 0.5
                print("-Aggiornamento Learning rate G2: ", self.opt_G2.lr.numpy())
                print("-Aggiornamento Learning rate D: ", self.opt_D.lr.numpy())
                print("")

            # Download from google colab
            if self.config.run_google_colab and (
                    epoch % self.config.download_weight == self.config.download_weight - 1):
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar logs/ -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar ./results_ssim -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar weights/ -idq')
                os.system('rm -r weights/*.hdf5')
                print("-RAR creato\n")

            print("#######")

    """
    Questo metodo esegue un training alternato tra generatore e discriminatore. Dapprima viene allenato il generatore e successivamente il discriminatore
    """

    def _train_step(self, train_it, epoch, id_batch):

        batch = next(train_it)
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

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)  # [batch, 96, 128, 15]
        output_G1 = self.model_G1(input_G1)  # output_g1 --> [batch, 96, 128, 1]
        output_G1 = tf.cast(output_G1, dtype=tf.float16)

        with tf.GradientTape() as g2_tape:

            # G2
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2 = self.model_G2(input_G2)  # [batch, 96, 128, 1]

            # Predizione D
            output_G2 = tf.cast(output_G2, dtype=tf.float16)
            refined_result = output_G1 + output_G2  # [batch, 96, 128, 1]
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                                axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = self.model_D(input_D)  # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1])  # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

            # Loss G2
            loss_value_G2 = G2.Loss(D_neg_refined_result, refined_result, image_raw_1, image_raw_0, mask_1, mask_0)

        self.opt_G2.minimize(loss_value_G2, var_list=self.model_G2.trainable_weights, tape=g2_tape)

        # if (id_batch + 1) % 2 == 1 and self.cnt_d == 0:
        #     print("G")
        #     #backprop G2
        #     self.opt_G2.minimize(loss_value_G2, var_list=self.model_G2.trainable_weights, tape=g2_tape)

        with tf.GradientTape() as d_tape:

            # Predizione G2
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2 = self.model_G2(input_G2)  # [batch, 96, 128, 1]

            # D
            output_G2 = tf.cast(output_G2, dtype=tf.float16)
            refined_result = output_G1 + output_G2  # [batch, 96, 128, 1]
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                                axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = self.model_D(input_D)  # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1])  # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

            # Loss D
            loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_pos_image_raw_1, D_neg_refined_result,
                                                                    D_neg_image_raw_0)

        self.opt_D.minimize(loss_value_D, var_list=self.model_D.trainable_weights, tape=d_tape)

        # if self.cnt_d == 0:
        #     if (id_batch + 1) % 2 == 0:
        #         # backprop D
        #         self.opt_D.minimize(loss_value_D, var_list=self.model_D.trainable_weights, tape=d_tape)
        # else:
        #     self.opt_D.minimize(loss_value_D, var_list=self.model_D.trainable_weights, tape=d_tape)


        # Metrics
        # - SSIM
        ssim_value = G2.m_ssim(refined_result, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G2.mask_ssim(refined_result, image_raw_1, mask_1, mean_0, mean_1)

        # - Real predette di refined_result dal discriminatore
        np_array_D_neg_refined_result = D_neg_refined_result.numpy()
        real_predette_refined_result_train = np_array_D_neg_refined_result[np_array_D_neg_refined_result > 0]

        # - Real predette di image_raw_0 dal discriminatore
        np_array_D_neg_image_raw_0 = D_neg_image_raw_0.numpy()
        real_predette_image_raw_0_train = np_array_D_neg_image_raw_0[np_array_D_neg_image_raw_0 > 0]

        # - Real predette di image_raw_1 (Target) dal discriminatore
        np_array_D_pos_image_raw_1 = D_pos_image_raw_1.numpy()
        real_predette_image_raw_1_train = np_array_D_pos_image_raw_1[np_array_D_pos_image_raw_1 > 0]

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_train == self.config.save_grid_ssim_epoch_train - 1:
            name_directory = os.path.join("./results_ssim/train", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G2_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         ssim=ssim_value,
                                         mask_ssim=mask_ssim_value))
            refined_result = utils_wgan.unprocess_image(refined_result, mean_0, 32765.5)
            grid.save_image(refined_result,
                            name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

            stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
            stack_pairs = np.array(
                [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
                 stack_pairs])
            txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
            file = open(name_directory + '/' + 'G2_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                                batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               real_predette_refined_result_train.shape[0], real_predette_image_raw_0_train.shape[0], \
               real_predette_image_raw_1_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy()

    def _valid_step(self, valid_it, epoch, id_batch):

        batch = next(valid_it)
        image_raw_0 = batch[0]  # [batch, 96,128, 1]
        image_raw_1 = batch[1]  # [batch, 96,128, 1]
        pose_1 = batch[2]  # [batch, 96,128, 1]
        mask_1 = batch[3]  # [batch, 96,128, 1]
        mask_0 = batch[4]  # [batch, 96,128, 1]
        pz_0 = batch[5]  # [batch, 1]
        pz_1 = batch[6]  # [batch, 1]
        name_0 = batch[7]  # [batch, 1]
        name_1 = batch[8]  # [batch, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)  # [batch, 96, 128, 1]
        output_G1 = self.model_G1(input_G1)  # output_g1 --> [batch, 96, 128, 1]

        # G2
        output_G1 = tf.cast(output_G1, dtype=tf.float16)
        input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
        output_G2 = self.model_G2(input_G2)  # [batch, 96, 128, 1]

        # D
        output_G2 = tf.cast(output_G2, dtype=tf.float16)
        refined_result = output_G1 + output_G2  # [batch, 96, 128, 1]
        input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                            axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
        output_D = self.model_D(input_D)  # [batch * 3, 1]
        output_D = tf.reshape(output_D, [-1])  # [batch*3]
        output_D = tf.cast(output_D, dtype=tf.float16)
        D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

        # Loss
        loss_value_G2 = G2.Loss(D_neg_refined_result, refined_result, image_raw_1, image_raw_0, mask_1, mask_0)
        loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_pos_image_raw_1, D_neg_refined_result,
                                                                D_neg_image_raw_0)

        # Metrics
        # - SSIM
        ssim_value = G2.m_ssim(refined_result, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G2.mask_ssim(refined_result, image_raw_1, mask_1, mean_0, mean_1)

        # - Real predette di refined_result dal discriminatore
        np_array_D_neg_refined_result = D_neg_refined_result.numpy()
        real_predette_refined_result_train = np_array_D_neg_refined_result[np_array_D_neg_refined_result > 0]

        # - Real predette di image_raw_0 dal discriminatore
        np_array_D_neg_image_raw_0 = D_neg_image_raw_0.numpy()
        real_predette_image_raw_0_train = np_array_D_neg_image_raw_0[np_array_D_neg_image_raw_0 > 0]

        # - Real predette di image_raw_1 (Target) dal discriminatore
        np_array_D_pos_image_raw_1 = D_pos_image_raw_1.numpy()
        real_predette_image_raw_1_train = np_array_D_pos_image_raw_1[np_array_D_pos_image_raw_1 > 0]

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_valid == self.config.save_grid_ssim_epoch_valid - 1:
            name_directory = os.path.join("./results_ssim/valid", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G2_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         ssim=ssim_value,
                                         mask_ssim=mask_ssim_value))
            refined_result = utils_wgan.unprocess_image(refined_result, mean_0, 32765.5)
            grid.save_image(refined_result,
                            name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

            stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
            stack_pairs = np.array(
                [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
                 stack_pairs])
            txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
            file = open(name_directory + '/' + 'G2_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                                batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G2.numpy (), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               real_predette_refined_result_train.shape[0], real_predette_image_raw_0_train.shape[0], \
               real_predette_image_raw_1_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy()


if __name__ == "__main__":
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    config.print_info()

    pg2 = PG2(config)  # Pose Guided ^2 network

    if config.trainig_G1:
        pg2.train_G1()
    elif config.trainig_GAN:
        pg2.train_conditional_GAN()
