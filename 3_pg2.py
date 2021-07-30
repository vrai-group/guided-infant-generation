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
from model import G1, G2, Discriminator
from datasets.BabyPose import BabyPose

class PG2(object):

    def __init__(self, config):
        # Trainer.__init__(self, config, data_loader=None)
        self.config = config
        self.babypose_obj = BabyPose(config)

    def train_G1(self):

        # Preprocess Dataset train
        dataset_train = self.babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_train)
        dataset_train = dataset_train.shuffle(self.config.dataset_train_len, reshuffle_each_iteration=True)
        dataset_train = self.babypose_obj.get_preprocess_G1_dataset(dataset_train)
        dataset_train = dataset_train.repeat(self.config.epochs)
        dataset_train = dataset_train.batch(self.config.batch_size_train)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch
        train_it = iter(dataset_train)

        # Preprocess Dataset valid
        dataset_valid = self.babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_valid)
        dataset_valid = self.babypose_obj.get_preprocess_G1_dataset(dataset_valid)
        # dataset_valid = dataset_valid.shuffle(self.config.dataset_valid_len, reshuffle_each_iteration=True)
        dataset_valid = dataset_valid.batch(self.config.batch_size_valid)
        dataset_valid = dataset_valid.repeat(self.config.epochs)
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)
        valid_it = iter(dataset_valid)

        # Costruzione modello
        model_g1 = G1.build_model(self.config)
        #model_g1.load_weights(os.path.join(self.config.weigths_path, ''))
        #model_g1.summary()

        # CallBacks
        filepath = os.path.join(self.config.weigths_path, 'Model_G1_epoch_{epoch:03d}-loss_{loss:2f}-mse_{mse:2f}-ssim_{m_ssim:2f}-mask_ssim_{mask_ssim:2f}-val_loss_{val_loss:2f}-val_mse_{val_mse:2f}-val_ssim_{val_m_ssim:2f}_val_mask_ssim_{val_mask_ssim:2f}.hdf5')
        checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, period=1)

        learning_rate_decay = LearningRateScheduler(G1.step_decay)

        model_g1.fit(train_it,
                     epochs=self.config.epochs,  # 100
                     steps_per_epoch=int(self.config.dataset_train_len / self.config.batch_size_train),  # 25600/4
                     callbacks=[checkPoint, learning_rate_decay],
                     validation_data=valid_it,
                     validation_steps=int(self.config.dataset_valid_len / self.config.batch_size_valid)
                     )

    def train_conditional_GAN(self):

        #Note: G1 è preaddestrato

        # Preprocess Dataset train
        dataset_train = self.babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_train)
        dataset_train = dataset_train.shuffle(self.config.dataset_train_len, reshuffle_each_iteration=True)
        dataset_train = self.babypose_obj.get_preprocess_GAN_dataset(dataset_train)
        dataset_train = dataset_train.batch(self.config.batch_size_train)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

        #Preprocess Dataset valid
        dataset_valid = self.babypose_obj.get_unprocess_dataset(config.data_tfrecord_path, config.name_tfrecord_valid)
        dataset_valid = self.babypose_obj.get_preprocess_GAN_dataset(dataset_valid)
        dataset_valid = dataset_valid.batch(self.config.batch_size_valid)
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)

        # Carico il modello preaddestrato G1
        self.model_G1 = G1.build_model(self.config)
        self.model_G1.load_weights(os.path.join(self.config.weigths_path, 'Model_G1_epoch_200-loss_0.000185-mse_inf-ssim_0.094355-mask_ssim_0.811881-val_loss_0.000289-val_mse_inf-val_ssim_0.027771_val_mask_ssim_0.799022.hdf5'))

        # Buildo la GAN
        # G2
        self.model_G2 = G2.build_model(self.config) # architettura Generatore G2
        #self.model_G2.summary()
        #self.model_G2.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_0.646448_real_valid_13_real_train_2790.hdf5'))
        self.opt_G2 = G2.optimizer() # ottimizzatore

        # D
        self.model_D = Discriminator.build_model(self.config)
        #self.model_D.summary()
        #self.model_D.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_2.855738_real_valid_13_real_train_2790.hdf5'))
        self.opt_D = Discriminator.optimizer()

        num_batches_train = int(self.config.dataset_train_len / self.config.batch_size_train)  # numero di batches nel dataset di train
        num_batches_valid = int(self.config.dataset_valid_len / self.config.batch_size_valid)  # numero di batches nel dataset di valid

        # Logs da salvare nella cartella logs per ogni epoca
        if os.path.exists(os.path.join(config.logs_path, 'logs_loss_train_G2.npy')):
            # Se esistenti, precarico i logs
            logs_loss_train_G2 = np.empty((self.config.epochs))
            logs_loss_train_D = np.empty((self.config.epochs))
            logs_loss_train_D_fake = np.empty((self.config.epochs))
            logs_loss_train_D_real = np.empty((self.config.epochs))

            a = np.load('logs_loss_train_G2.npy')
            num = a.shape[0]
            logs_loss_train_G2[:num] = np.load('logs_loss_train_G2.npy')
            logs_loss_train_D[:num] = np.load('logs_loss_train_D_total.npy')
            logs_loss_train_D_fake[:num] = np.load('logs_loss_train_D_fake.npy')
            logs_loss_train_D_real[:num] = np.load('logs_loss_train_D_fake.npy')

        else:
            # Altrimenti li creo ex novo
            logs_loss_train_G2 = np.empty((self.config.epochs))
            logs_loss_train_D = np.empty((self.config.epochs))
            logs_loss_train_D_fake = np.empty((self.config.epochs))
            logs_loss_train_D_real = np.empty((self.config.epochs))


        for epoch in range(self.config.epochs):
            train_it = iter(dataset_train)  # rinizializzo l iteratore sul train dataset
            valid_it = iter(dataset_valid)  # rinizializzo l iteratore sul valid dataset

            mean_loss_G2_train = 0  # calcolo la media della loss ad ogni iterazione sul batch
            mean_loss_D_train = 0  # calcolo la media  della loss ad ogni iterazione sul batch
            mean_loss_D_train_fake = 0
            mean_loss_D_train_real = 0
            mean_ssim_train = 0
            mean_mask_ssim_train = 0
            mean_ssim_valid = 0
            mean_mask_ssim_valid = 0
            cnt_real_predette_train = 0  # counter per le reali (output_G1 + output_G2) predette nel train
            cnt_real_predette_valid = 0 # counter per le reali (output_G1 + output_G2) predette nel valid

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

            # Train
            for id_batch in range(num_batches_train):
                loss_values_train_G2[id_batch], loss_values_train_D[id_batch], \
                loss_values_train_D_fake[id_batch], loss_values_train_D_real[id_batch], \
                real_predette_train, ssim_train[id_batch], mask_ssim_train[id_batch] = self._train_step(train_it, epoch, id_batch)

                # Calcolo media
                # Loss
                mean_loss_G2_train = np.mean(loss_values_train_G2[:id_batch+1])
                mean_loss_D_train = np.mean(loss_values_train_D[:id_batch+1])
                mean_loss_D_train_fake = np.mean(loss_values_train_D_fake[:id_batch+1])
                mean_loss_D_train_real = np.mean(loss_values_train_D_real[:id_batch+1])
                # Metrics
                cnt_real_predette_train += real_predette_train
                mean_ssim_train = np.mean(ssim_train[:id_batch+1])
                mean_mask_ssim_train = np.mean(mask_ssim_train[:id_batch+1])

                # Logs a schermo
                sys.stdout.write('\r')
                sys.stdout.write('Epoch {epoch} step {id_batch} / {num_batches} --> loss_G2: {loss_G2:2.3f}, '
                                 'loss_D: {loss_D:2f}, loss_D_fake: {loss_D_fake:2f}, loss_D_real: {loss_D_real:2f},'
                                 ' ssmi: {ssmi:2f}, mask_ssmi: {mask_ssmi:2f}, real_predette: {real:1} / {total_train}'.format(epoch=epoch + 1,
                                 id_batch=id_batch, num_batches=num_batches_train, loss_G2=mean_loss_G2_train,
                                 loss_D=mean_loss_D_train, loss_D_fake=mean_loss_D_train_fake, loss_D_real=mean_loss_D_train_real,
                                 real=cnt_real_predette_train, ssmi=mean_ssim_train, mask_ssmi=mean_mask_ssim_train, total_train=self.config.dataset_train_len))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.write('Validazione..')
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                 loss_values_valid_G2[id_batch], loss_values_valid_D[id_batch], \
                 loss_values_valid_D_fake[id_batch], loss_values_valid_D_real[id_batch], \
                 real_predette_valid, ssim_valid[id_batch], mask_ssim_valid[id_batch] = self._valid_step(valid_it, epoch, id_batch)

                 # Metrics
                 cnt_real_predette_valid += real_predette_valid

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
                             'val_loss_D_real: {loss_D_real:2f}, val_ssmi: {ssmi:2f}, val_mask_ssmi: {mask_ssmi:2f},'
                             'val_real_predette: {real:1} / {total_valid}'.format(
                loss_G2=mean_loss_G2_valid,
                loss_D=mean_loss_D_valid, loss_D_fake=mean_loss_D_valid_fake, loss_D_real=mean_loss_D_valid_real,
                real=cnt_real_predette_valid, ssmi=mean_ssim_valid, mask_ssmi=mean_mask_ssim_valid,
                total_valid=self.config.dataset_valid_len))
            sys.stdout.flush()
            sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.write('\n')

            # Save weights
            name_model = "Model_G2_epoch_{epoch:03d}-loss_{loss:2f}-ssmi_{ssmi:2f}-mask_ssmi_{mask_ssim:2f}-real_train_{real_train}" \
                         "-val_loss_{val_loss}-val_ssim_{val_ssim:2f}-val_mask_ssim_{val_mask_ssim:2f}-real_valid_{real_valid}.hdf5".format(
                          epoch=epoch+1, loss=mean_loss_G2_train, ssmi=mean_ssim_train, mask_ssim=mean_mask_ssim_train,  real_train=cnt_real_predette_train,
                          val_loss=mean_loss_G2_valid, val_ssim=mean_ssim_valid, val_mask_ssim=mean_mask_ssim_valid, real_valid=cnt_real_predette_valid )
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_G2.save_weights(filepath)

            # loss_values_valid_D[id_batch], \
            # loss_values_valid_D_fake[id_batch], loss_values_valid_D_real
            name_model = "Model_D_epoch_{epoch:03d}-" \
                         "loss_{loss:2f}-" \
                         "loss_values_D_fake{loss_D_fake:2f}-" \
                         "loss_values_D_real{loss_D_real:2f}-" \
                         "real_train_{real_train}-" \
                         "val_loss_{val_loss:2f}-" \
                         "val_loss_values_D_fake{val_loss_D_real}-" \
                         "val_loss_values_D_real{val_loss_D_fake}-" \
                         "val_real_valid_{real_valid}-real_valid_{real_valid}.hdf5".format(
                          epoch=epoch+1, loss=mean_loss_D_train, loss_D_fake=mean_loss_D_train_fake, loss_D_real=mean_loss_D_train_real,
                          real_train=cnt_real_predette_train, val_loss=mean_loss_D_valid, val_loss_D_real=mean_loss_D_valid_real, val_loss_D_fake=mean_loss_D_valid_fake, real_valid=cnt_real_predette_valid,
                          )
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_D.save_weights(filepath)

            #Save logs
            logs_loss_train_G2[epoch] = mean_loss_G2_train
            logs_loss_train_D[epoch] = mean_loss_D_train
            logs_loss_train_D_fake[epoch] = mean_loss_D_train_fake
            logs_loss_train_D_real[epoch] = mean_loss_D_train_real

            np.save(os.path.join(self.config.logs_path,'logs_loss_train_G2.npy'), logs_loss_train_G2[:epoch+1])
            np.save(os.path.join(self.config.logs_path,'logs_loss_train_D_total.npy'), logs_loss_train_D[:epoch+1])
            np.save(os.path.join(self.config.logs_path,'logs_loss_train_D_fake.npy'), logs_loss_train_D_fake[:epoch+1])
            np.save(os.path.join(self.config.logs_path,'logs_loss_train_D_real.npy'), logs_loss_train_D_real[:epoch+1])

            # Update learning rate
            if epoch % self.config.lr_update_epoch == self.config.lr_update_epoch - 1:
                self.opt_G2.lr = self.opt_G2.lr * 0.5
                self.opt_D.lr = self.opt_D.lr * 0.5
                print("-Aggiornamento Learning rate G2: ", self.opt_G2.lr.numpy())
                print("-Aggiornamento Learning rate D: ", self.opt_D.lr.numpy())
                print("")

            # Download from google colab
            if self.config.run_google_colab and (epoch % self.config.download_weight == self.config.download_weight-1):
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar logs/ -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar ./results_ssim -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar weights/ -idq')
                os.system('rm -r weights/*.hdf5')
                print("-RAR creato\n")

            print("#######")

    def _train_step(self, train_it, epoch, id_batch):

        batch = next(train_it)
        image_raw_0 = batch[0] #[batch, 96, 128, 1]
        image_raw_1 = batch[1] #[batch, 96,128, 1]
        pose_1 = batch[2] #[batch, 96,128, 14]
        mask_1 = batch[3] #[batch, 96,128, 1]
        mask_0 = batch[4]  # [batch, 96,128, 1]
        pz_0 = batch[5]  # [batch, 1]
        pz_1 = batch[6]  # [batch, 1]
        name_0 = batch[7]  # [batch, 1]
        name_1 = batch[8]  # [batch, 1]

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1) # [batch, 96, 128, 15]
        output_G1 = self.model_G1(input_G1) # output_g1 --> [batch, 96, 128, 1]

        with tf.GradientTape() as g2_tape, tf.GradientTape() as d_tape:

            # G2
            output_G1 = tf.cast(output_G1, dtype=tf.float16)
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2 = self.model_G2(input_G2) # [batch, 96, 128, 1]

            # D
            output_G2 = tf.cast(output_G2, dtype=tf.float16)
            refined_result = output_G1 + output_G2 # [batch, 96, 128, 1]
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0], axis=0) # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = self.model_D(input_D) # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1]) # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3) # [batch]

            # Loss
            loss_value_G2 = G2.Loss(D_neg_refined_result, refined_result, image_raw_1, image_raw_0, mask_1, mask_0)
            loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0)

            # metric
            # Real predette dal discriminatore
            np_array_D_neg_refined_result = D_neg_refined_result.numpy()
            real_predette_train = np_array_D_neg_refined_result[np_array_D_neg_refined_result > 0]

            # SSIM
            ssim_value = G2.m_ssim(refined_result, image_raw_0)
            mask_ssim_value = G2.mask_ssim(refined_result, image_raw_1, mask_1)

        # backprop
        self.opt_G2.minimize(loss_value_G2, var_list=self.model_G2.trainable_weights, tape=g2_tape)
        self.opt_D.minimize(loss_value_D, var_list=self.model_D.trainable_weights, tape=d_tape)

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_train == self.config.save_grid_ssim_epoch_train - 1:
            name_directory = os.path.join("./results_ssim/train", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory, 'G2_ssim_epoch_{epoch}_batch_{batch}_mask_ssim{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         mask_ssim=mask_ssim_value))
            refined_result = utils_wgan.unprocess_image(refined_result, 350, 32765.5)
            grid.save_image(refined_result, name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

            stack_pz = np.c_[pz_0.numpy(), pz_1.numpy()]
            stack_pz = np.array([[p[0].decode('utf-8'), p[1].decode('utf-8')] for p in stack_pz])
            stack_im = np.c_[name_0.numpy(), name_1.numpy()]
            stack_im = np.array([[p[0].decode('utf-8'), p[1].decode('utf-8')] for p in stack_im])
            txt_file = 'pz_pair: \n\n {stack_pz} \n\n\n ' \
                       '########'\
                       'im_pair: \n\n {stack_im}'.format(
                stack_pz=np.array2string(stack_pz),
                stack_im=np.array2string(stack_im))
            file = open(name_directory + '/' + 'G2_ssim_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,  batch=id_batch), "w")
            file.write(txt_file)
            file.close()


        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), real_predette_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy()

    def _valid_step(self, valid_it, epoch, id_batch):

        batch = next(valid_it)
        image_raw_0 = batch[0] #[batch, 96,128, 1]
        image_raw_1 = batch[1] #[batch, 96,128, 1]
        pose_1 = batch[2] #[batch, 96,128, 1]
        mask_1 = batch[3] #[batch, 96,128, 1]
        mask_0 = batch[4]  # [batch, 96,128, 1]
        pz_0 = batch[5]  # [batch, 1]
        pz_1 = batch[6]  # [batch, 1]
        name_0 = batch[7]  # [batch, 1]
        name_1 = batch[8]  # [batch, 1]

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1) # [batch, 96, 128, 1]
        output_G1 = self.model_G1(input_G1) # output_g1 --> [batch, 96, 128, 1]

        # G2
        output_G1 = tf.cast(output_G1, dtype=tf.float16)
        input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
        output_G2 = self.model_G2(input_G2)  # [batch, 96, 128, 1]

        # D
        output_G2 = tf.cast(output_G2, dtype=tf.float16)
        refined_result = output_G1 + output_G2  # [batch, 96, 128, 1]
        input_D = tf.concat([image_raw_1, refined_result, image_raw_0], axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
        output_D = self.model_D(input_D)  # [batch * 3, 1]
        output_D = tf.reshape(output_D, [-1])  # [batch*3]
        output_D = tf.cast(output_D, dtype=tf.float16)
        D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

        # Loss
        loss_value_G2 = G2.Loss(D_neg_refined_result, refined_result, image_raw_1, image_raw_0, mask_1, mask_0)
        loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_pos_image_raw_1, D_neg_refined_result,
                                                                D_neg_image_raw_0)

        # Metrics
        real_predette = D_neg_refined_result[D_neg_refined_result > 0]
        # SSIM
        ssim_value = G2.m_ssim(refined_result, image_raw_0)
        mask_ssim_value = G2.mask_ssim(refined_result, image_raw_1, mask_1)

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_valid == self.config.save_grid_ssim_epoch_valid - 1:
            name_directory = os.path.join("./results_ssim/valid", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory, 'G2_ssim_epoch_{epoch}_batch_{batch}_mask_ssim{mask_ssim}.png'.format(
                epoch=epoch + 1,
                batch=id_batch,
                mask_ssim=mask_ssim_value))
            refined_result = utils_wgan.unprocess_image(refined_result, 350, 32765.5)
            grid.save_image(refined_result, name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

            stack_pz = np.c_[pz_0.numpy(), pz_1.numpy()]
            stack_pz = np.array([[p[0].decode('utf-8'), p[1].decode('utf-8')] for p in stack_pz])
            stack_im = np.c_[name_0.numpy(), name_1.numpy()]
            stack_im = np.array([[p[0].decode('utf-8'), p[1].decode('utf-8')] for p in stack_im])
            txt_file = 'pz_pair: \n\n {stack_pz} \n\n\n ' \
                       '######## \n\n' \
                       'im_pair: \n\n {stack_im}'.format(
                stack_pz=np.array2string(stack_pz),
                stack_im=np.array2string(stack_im))
            file = open(name_directory+'/'+ 'G2_ssim_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1, batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), real_predette.shape[0], ssim_value.numpy(), mask_ssim_value.numpy()

if __name__ == "__main__":
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()

    pg2 = PG2(config)  # Pose Guided ^2 network

    #pg2.train_G1()
    pg2.train_conditional_GAN()

        




