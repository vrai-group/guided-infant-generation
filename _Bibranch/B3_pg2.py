"""
Questo script consente di avviare il training del G1 e della GAN
"""
import os
import sys
import numpy as np
import tensorflow as tf

from utils import augumentation, grid, utils_wgan
from models import G1_Bibranch, G2_Bibranch, Discriminator
from _Bibranch.datasets.BabyPose import BabyPose


class PG2(object):

    def __init__(self, config):
        self.config = config
        self.babypose_obj = BabyPose()



    def train_G1_Bibranch(self):

        # -Caricamento datascaet
        dataset_train = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_train)
        dataset_train = dataset_train.batch(1)

        dataset_valid = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_valid)
        dataset_valid = dataset_valid.batch(1)

        # -Costruzione modello
        self.model_G1_Bibranch = G1_Bibranch.build_model()
        self.opt_G1_Bibranch = G1_Bibranch.optimizer()
        self.model_G1_Bibranch.summary()

        # -History del training
        history_G1_Bibranch = {'epoch': 0,
                      'loss_train': np.empty((self.config.epochs_G1_Bibranch)),
                      'ssim_train': np.empty((self.config.epochs_G1_Bibranch)),
                      'mask_ssim_train': np.empty((self.config.epochs_G1_Bibranch)),

                      'loss_valid': np.empty((self.config.epochs_G1_Bibranch)),
                      'ssim_valid': np.empty((self.config.epochs_G1_Bibranch)),
                      'mask_ssim_valid': np.empty((self.config.epochs_G1_Bibranch))}

        # Se esistenti, precarico i logs
        if os.path.exists(os.path.join(config.logs_path, 'history_G1_Bibranch.npy')):
            old_history_G1_Bibranch = np.load(os.path.join(self.config.logs_path, 'history_G1_Bibranch.npy'), allow_pickle='TRUE')
            epoch = old_history_G1_Bibranch.item().get('epoch')
            for key, value in old_history_G1_Bibranch.item().items():
                if key == 'epoch':
                    history_G1_Bibranch[key] = value
                else:
                    history_G1_Bibranch[key][:epoch] = value[:epoch]

        for epoch in range(history_G1_Bibranch['epoch'], self.config.epochs_G1_Bibranch):
            train_it = iter(dataset_train)
            valid_it = iter(dataset_valid)

            ## Augumentazione Dataset
            name_tfrecord_aug_train, dataset_train_aug_len = augumentation.apply_augumentation(train_it, config, "train")
            name_tfrecord_aug_valid, dataset_valid_aug_len = augumentation.apply_augumentation(valid_it, config, "valid")

            print("\nAugumentazione terminata: ")
            print("- lunghezza train: ", dataset_train_aug_len)
            print("- lunghezza valid: ", dataset_valid_aug_len)
            print("\n")

            ## Preprocess Dataset Augumentato
            dataset_train_aug = self.babypose_obj.get_unprocess_dataset(name_tfrecord_aug_train)
            dataset_train_aug = dataset_train_aug.shuffle(dataset_train_aug_len, reshuffle_each_iteration=True)
            dataset_train_aug = self.babypose_obj.get_preprocess_G1_Bibranch_dataset(dataset_train_aug)
            dataset_train_aug = dataset_train_aug.batch(self.config.batch_size_train)
            dataset_train_aug = dataset_train_aug.prefetch(tf.data.AUTOTUNE)

            dataset_valid_aug = self.babypose_obj.get_unprocess_dataset(name_tfrecord_aug_valid)
            dataset_valid_aug = self.babypose_obj.get_preprocess_G1_Bibranch_dataset(dataset_valid_aug)
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
                logs_to_print['loss_values_train'][id_batch], logs_to_print['ssim_train'][id_batch], \
                logs_to_print['mask_ssim_train'][id_batch] = self._train_step_G1_Bibranch(train_it, epoch, id_batch)

                # Logs a schermo
                sys.stdout.write('\r')
                sys.stdout.write('Epoch {epoch} step {id_batch} / {num_batches} --> loss_G1_Bibranch: {loss_G1_Bibranch:.4f}, '
                                 'ssmi: {ssmi:.4f}, mask_ssmi: {mask_ssmi:.4f}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch + 1,
                    num_batches=num_batches_train,
                    loss_G1_Bibranch=np.mean(logs_to_print['loss_values_train'][:id_batch + 1]),
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
                    = self._valid_step_G1_Bibranch(valid_it, epoch, id_batch)

                sys.stdout.write('\r')
                sys.stdout.write('{id_batch} / {total}'.format(id_batch=id_batch + 1, total=num_batches_valid))
                sys.stdout.flush()

            sys.stdout.write('\r\r')
            sys.stdout.write('val_loss_G1_Bibranch: {loss_G1_Bibranch:.4f}, val_ssmi: {ssmi:.4f}, val_mask_ssmi: {mask_ssmi:.4f}'.format(
                loss_G1_Bibranch=np.mean(logs_to_print['loss_values_valid']),
                ssmi=np.mean(logs_to_print['ssim_valid']),
                mask_ssmi=np.mean(logs_to_print['mask_ssim_valid'])))
            sys.stdout.flush()
            sys.stdout.write('\n\n')

            # -CallBacks
            # --Save weights
            name_model = 'Model_G1_Bibranch_epoch_{epoch:03d}-' \
                         'loss_{loss:.3f}-' \
                         'ssim_{m_ssim:.3f}-' \
                         'mask_ssim_{mask_ssim:.3f}-' \
                         'val_loss_{val_loss:.3f}-' \
                         'val_ssim_{val_m_ssim:.3f}-' \
                         'val_mask_ssim_{val_mask_ssim:.3f}.hdf5'.format(
                epoch=epoch + 1,
                loss=np.mean(logs_to_print['loss_values_train']),
                m_ssim=np.mean(logs_to_print['ssim_train']),
                mask_ssim=np.mean(logs_to_print['mask_ssim_train']),
                val_loss=np.mean(logs_to_print['loss_values_valid']),
                val_m_ssim=np.mean(logs_to_print['ssim_valid']),
                val_mask_ssim=np.mean(logs_to_print['mask_ssim_valid']))
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_G1_Bibranch.save_weights(filepath)

            # --Update learning rate
            if epoch % self.config.lr_update_epoch_G1_Bibranch == self.config.lr_update_epoch_G1_Bibranch - 1:
                self.opt_G1_Bibranch.lr = self.opt_G1_Bibranch.lr * self.config.drop_rate_G1_Bibranch
                print("-Aggiornamento Learning rate G1_Bibranch: ", self.opt_G1_Bibranch.lr.numpy())
                print("\n")

            # --Save logs
            history_G1_Bibranch['epoch'] = epoch + 1
            history_G1_Bibranch['loss_train'][epoch] = np.mean(logs_to_print['loss_values_train'])
            history_G1_Bibranch['ssim_train'][epoch] = np.mean(logs_to_print['ssim_train'])
            history_G1_Bibranch['mask_ssim_train'][epoch] = np.mean(logs_to_print['mask_ssim_train'])
            history_G1_Bibranch['loss_valid'][epoch] = np.mean(logs_to_print['loss_values_valid'])
            history_G1_Bibranch['ssim_valid'][epoch] = np.mean(logs_to_print['ssim_valid'])
            history_G1_Bibranch['mask_ssim_valid'][epoch] = np.mean(logs_to_print['mask_ssim_valid'])
            np.save(os.path.join(self.config.logs_path, 'history_G1_Bibranch.npy'), history_G1_Bibranch)

            # --Save Gooogle colab
            if self.config.run_google_colab and (
                    epoch % self.config.download_weight == self.config.download_weight - 1):
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar ./logs -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar ./results_ssim -idq')
                os.system('rar a /gdrive/MyDrive/weights_and_logs.rar weights/ -idq')
                print("-RAR creato\n")
                # Pulizia enviroment
                os.system('rm -r weights/*.hdf5')
                os.system('rm -r results_ssim/G1_Bibranch/train results_ssim/G1_Bibranch/valid')
                os.system('rm -r results_ssim/G1_Bibranch/train results_ssim/G1_Bibranch/valid')
                os.system('mkdir results_ssim/G1_Bibranch/train results_ssim/G1_Bibranch/valid')
                print("-Pulizia enviroment eseguita\n")

        print("#############\n\n")

    def _train_step_G1_Bibranch(self, train_it, epoch, id_batch):

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

            # G1_Bibranch
            input_G1_Bibranch = tf.concat([image_raw_0, pose_1], axis=-1)
            output_G1_Bibranch = self.model_G1_Bibranch(input_G1_Bibranch)  # [batch, 96, 128, 1] dtype=float32

            # Loss G1_Bibranch
            loss_value_G1_Bibranch = G1_Bibranch.PoseMaskLoss1(output_G1_Bibranch, image_raw_1, mask_1)

        self.opt_G1_Bibranch.minimize(loss_value_G1_Bibranch, var_list=self.model_G1_Bibranch.trainable_weights, tape=g1_tape)

        # Metrics
        # - SSIM
        ssim_value = G1_Bibranch.m_ssim(output_G1_Bibranch, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G1_Bibranch.mask_ssim(output_G1_Bibranch, image_raw_1, mask_1, mean_0, mean_1)

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_train == self.config.save_grid_ssim_epoch_train - 1:
            name_directory = os.path.join("./results_ssim/G1_Bibranch/train", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G1_Bibranch_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         ssim=ssim_value,
                                         mask_ssim=mask_ssim_value))
            mean_0 = tf.cast(mean_0, dtype=tf.float32)
            output_G1_Bibranch = utils_wgan.unprocess_image(output_G1_Bibranch, mean_0, 32765.5)
            grid.save_image(output_G1_Bibranch,
                            name_grid)  # si salva in una immagine contenente una griglia tutti i  G1_Bibranch + DiffMap

            stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
            stack_pairs = np.array(
                [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
                 stack_pairs])
            txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
            file = open(name_directory + '/' + 'G1_Bibranch_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                           batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G1_Bibranch, ssim_value, mask_ssim_value

    def _valid_step_G1_Bibranch(self, valid_it, epoch, id_batch):

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

        # G1_Bibranch
        input_G1_Bibranch = tf.concat([image_raw_0, pose_1], axis=-1)
        output_G1_Bibranch = self.model_G1_Bibranch(input_G1_Bibranch)  # [batch, 96, 128, 1] dtype=float32

        # Loss G1_Bibranch
        loss_value_G1_Bibranch = G1_Bibranch.PoseMaskLoss1(output_G1_Bibranch, image_raw_1, mask_1)

        # Metrics
        # - SSIM
        ssim_value = G1_Bibranch.m_ssim(output_G1_Bibranch, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G1_Bibranch.mask_ssim(output_G1_Bibranch, image_raw_1, mask_1, mean_0, mean_1)

        # Save griglia di immagini predette
        if epoch % self.config.save_grid_ssim_epoch_valid == self.config.save_grid_ssim_epoch_valid - 1:
            name_directory = os.path.join("./results_ssim/G1_Bibranch/valid", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G1_Bibranch_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                         epoch=epoch + 1,
                                         batch=id_batch,
                                         ssim=ssim_value,
                                         mask_ssim=mask_ssim_value))
            mean_0 = tf.cast(mean_0, dtype=tf.float32)
            output_G1_Bibranch = utils_wgan.unprocess_image(output_G1_Bibranch, mean_0, 32765.5)
            grid.save_image(output_G1_Bibranch,
                            name_grid)  # si salva in una immagine contenente una griglia tutti i  G1_Bibranch + DiffMap

            stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
            stack_pairs = np.array(
                [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
                 stack_pairs])
            txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
            file = open(name_directory + '/' + 'G1_Bibranch_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                           batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G1_Bibranch, ssim_value, mask_ssim_value

    def train_conditional_GAN(self):

        # Note: G1 è preaddestrato

        # -Caricamento dataset
        dataset_train = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_train)
        dataset_train = dataset_train.shuffle(self.config.dataset_train_len, reshuffle_each_iteration=True)
        dataset_train = self.babypose_obj.get_preprocess_G1_Bibranch_dataset(dataset_train)
        dataset_train = dataset_train.batch(self.config.batch_size_train)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

        dataset_valid = self.babypose_obj.get_unprocess_dataset(self.config.name_tfrecord_valid)
        dataset_valid = self.babypose_obj.get_preprocess_G1_Bibranch_dataset(dataset_valid)
        dataset_valid = dataset_valid.batch(self.config.batch_size_valid)
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)

        num_batches_train = self.config.dataset_train_len // self.config.batch_size_train
        num_batches_valid = self.config.dataset_valid_len // self.config.batch_size_valid


        # Carico il modello preaddestrato G1
        self.model_G1_Bibranch = G1_Bibranch.build_model()
        self.model_G1_Bibranch.load_weights(os.path.join(self.config.weigths_path,'Model_G1_Bibranch_epoch_005-loss_0.000-ssim_0.943-mask_ssim_0.984-val_loss_0.001-val_ssim_0.917-val_mask_ssim_0.979.hdf5'))
        self.model_G1_Bibranch.summary()

        # Buildo la GAN
        # G2_Bibranch
        self.model_G2_Bibranch = G2_Bibranch.build_model()  # architettura Generatore G2_Bibranch
        self.opt_G2_Bibranch = G2_Bibranch.optimizer()  # ottimizzatore

        # D
        self.model_D = Discriminator.build_model()
        self.opt_D = Discriminator.optimizer()

        # -History del training
        history_GAN = {'epoch': 0,
                       'loss_train_G2_Bibranch': np.empty((self.config.epochs_GAN)),
                       'loss_train_D': np.empty((self.config.epochs_GAN)),
                       'loss_train_fake_D': np.empty((self.config.epochs_GAN)),
                       'loss_train_real_D': np.empty((self.config.epochs_GAN)),
                       'ssim_train': np.empty((self.config.epochs_GAN)),
                       'mask_ssim_train': np.empty((self.config.epochs_GAN)),
                       'r_r_train': np.empty((self.config.epochs_GAN), dtype=np.uint32),
                       'img_0_train': np.empty((self.config.epochs_GAN), dtype=np.uint32),
                       'img_1_train': np.empty((self.config.epochs_GAN), dtype=np.uint32),

                       'loss_valid_G2_Bibranch': np.empty((self.config.epochs_GAN)),
                       'loss_valid_D': np.empty((self.config.epochs_GAN)),
                       'loss_valid_fake_D': np.empty((self.config.epochs_GAN)),
                       'loss_valid_real_D': np.empty((self.config.epochs_GAN)),
                       'ssim_valid': np.empty((self.config.epochs_GAN)),
                       'mask_ssim_valid': np.empty((self.config.epochs_GAN)),
                       'r_r_valid': np.empty((self.config.epochs_GAN), dtype=np.uint32),
                       'img_0_valid': np.empty((self.config.epochs_GAN), dtype=np.uint32),
                       'img_1_valid': np.empty((self.config.epochs_GAN), dtype=np.uint32),
                       }

        # Se esistenti, precarico i logs
        if os.path.exists(os.path.join(config.logs_path, 'history_GAN.npy')):
            old_history_GAN = np.load(os.path.join(self.config.logs_path, 'history_GAN.npy'), allow_pickle='TRUE')
            # epoch = old_history_G1[()]['epoch'] --> anche in questo modi riesco ad ottenere il value dell'epoca
            epoch = old_history_GAN.item().get('epoch')
            for key, value in old_history_GAN.item().items():
                if key == 'epoch':
                    history_GAN[key] = value
                else:
                    history_GAN[key][:epoch] = value[:epoch]

        for epoch in range(history_GAN['epoch'], self.config.epochs_GAN):
            train_it = iter(dataset_train)
            valid_it = iter(dataset_valid)

            ## Augumentazione Dataset
            # name_tfrecord_aug_train, dataset_train_aug_len = augumentation.apply_augumentation(train_it, config, "train")
            # name_tfrecord_aug_valid, dataset_valid_aug_len = augumentation.apply_augumentation(valid_it, config, "valid")

            # print("\nAugumentazione terminata: ")
            # print("- lunghezza train: ", dataset_train_aug_len)
            # print("- lunghezza valid: ", dataset_valid_aug_len)
            # print("\n")

            ## Preprocess Dataset Augumentato
            # dataset_train_aug = self.babypose_obj.get_unprocess_dataset(name_tfrecord_aug_train)
            # dataset_train_aug = dataset_train_aug.shuffle(dataset_train_aug_len, reshuffle_each_iteration=True)
            # dataset_train_aug = self.babypose_obj.get_preprocess_G1_dataset(dataset_train_aug)
            # dataset_train_aug = dataset_train_aug.batch(self.config.batch_size_train)
            # dataset_train_aug = dataset_train_aug.prefetch(tf.data.AUTOTUNE)

            # dataset_valid_aug = self.babypose_obj.get_unprocess_dataset(name_tfrecord_aug_valid)
            # dataset_valid_aug = self.babypose_obj.get_preprocess_G1_dataset(dataset_valid_aug)
            # dataset_valid_aug = dataset_valid_aug.batch(self.config.batch_size_valid)
            # dataset_valid_aug = dataset_valid_aug.prefetch(tf.data.AUTOTUNE)

            # numero di batches nel dataset
            # num_batches_train = dataset_train_aug_len // self.config.batch_size_train
            # num_batches_valid = dataset_valid_aug_len // self.config.batch_size_valid

            # train_it = iter(dataset_train_aug)  # rinizializzo l iteratore sul train dataset
            # valid_it = iter(dataset_valid_aug)  # rinizializzo l iteratore sul valid dataset

            # Vettori che mi serviranno per salvare i valori per ogni epoca in modo tale da printare a schermo le medie
            logs_to_print = {'loss_values_train_G2_Bibranch': np.empty((num_batches_train)),
                             'loss_values_train_D': np.empty((num_batches_train)),
                             'loss_values_train_fake_D': np.empty((num_batches_train)),
                             'loss_values_train_real_D': np.empty((num_batches_train)),
                             'ssim_train': np.empty((num_batches_train)),
                             'mask_ssim_train': np.empty((num_batches_train)),
                             'r_r_train': np.empty((num_batches_train), dtype=np.uint32),
                             'img_0_train': np.empty((num_batches_train), dtype=np.uint32),
                             'img_1_train': np.empty((num_batches_train), dtype=np.uint32),

                             'loss_values_valid_G2_Bibranch': np.empty((num_batches_valid)),
                             'loss_values_valid_D': np.empty((num_batches_valid)),
                             'loss_values_valid_fake_D': np.empty((num_batches_valid)),
                             'loss_values_valid_real_D': np.empty((num_batches_valid)),
                             'ssim_valid': np.empty((num_batches_valid)),
                             'mask_ssim_valid': np.empty((num_batches_valid)),
                             'r_r_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             'img_0_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             'img_1_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             }

            # Train
            for id_batch in range(num_batches_train):
                logs_to_print['loss_values_train_G2_Bibranch'][id_batch], logs_to_print['loss_values_train_D'][id_batch], \
                logs_to_print['loss_values_train_fake_D'][id_batch], logs_to_print['loss_values_train_real_D'][id_batch], \
                logs_to_print['r_r_train'][id_batch], logs_to_print['img_0_train'][id_batch], \
                logs_to_print['img_1_train'][id_batch], \
                logs_to_print['ssim_train'][id_batch], logs_to_print['mask_ssim_train'][id_batch] = self._train_step(
                    train_it, epoch, id_batch)

                # Logs a schermo
                sys.stdout.write('\r')
                sys.stdout.write('Epoch {epoch} step {id_batch} / {num_batches} --> loss_G2_Bibranch: {loss_G2_Bibranch:2f}, '
                                 'loss_D: {loss_D:2f}, loss_D_fake: {loss_D_fake:2f}, loss_D_real: {loss_D_real:2f}, '
                                 'ssmi: {ssmi:2f}, mask_ssmi: {mask_ssmi:2f}, '
                                 'real_predette:: r_r:{r_r:1}, im_0:{im_0:1}, im_1:{im_1:1} / {total_train}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch + 1,
                    num_batches=num_batches_train,
                    loss_G2_Bibranch=np.mean(logs_to_print['loss_values_train_G2_Bibranch'][:id_batch + 1]),
                    loss_D=np.mean(logs_to_print['loss_values_train_D'][:id_batch + 1]),
                    loss_D_fake=np.mean(logs_to_print['loss_values_train_fake_D'][:id_batch + 1]),
                    loss_D_real=np.mean(logs_to_print['loss_values_train_real_D'][:id_batch + 1]),
                    r_r=np.sum(logs_to_print['r_r_train'][:id_batch + 1]),
                    im_0=np.sum(logs_to_print['img_0_train'][:id_batch + 1]),
                    im_1=np.sum(logs_to_print['img_1_train'][:id_batch + 1]),
                    ssmi=np.mean(logs_to_print['ssim_train'][:id_batch + 1]),
                    mask_ssmi=np.mean(logs_to_print['mask_ssim_train'][:id_batch + 1]),
                    total_train=self.config.dataset_train_len))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.write('Validazione..')
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                logs_to_print['loss_values_valid_G2_Bibranch'][id_batch], logs_to_print['loss_values_valid_D'][id_batch], \
                logs_to_print['loss_values_valid_fake_D'][id_batch], logs_to_print['loss_values_valid_real_D'][
                    id_batch], \
                logs_to_print['r_r_valid'][id_batch], logs_to_print['img_0_valid'][id_batch], \
                logs_to_print['img_1_valid'][id_batch], \
                logs_to_print['ssim_valid'][id_batch], logs_to_print['mask_ssim_valid'][id_batch] = self._valid_step(
                    valid_it, epoch, id_batch)

                sys.stdout.write('\r')
                sys.stdout.write('{id_batch} / {total}'.format(id_batch=id_batch + 1, total=num_batches_valid))
                sys.stdout.flush()

            sys.stdout.write('\r\r')
            sys.stdout.write(
                'val_loss_G2_Bibranch: {loss_G2_Bibranch:.4f}, val_loss_D: {loss_D:.4f}, val_loss_D_fake: {loss_D_fake:.4f}, '
                'val_loss_D_real: {loss_D_real:.4f}, val_ssmi: {ssmi:.4f}, val_mask_ssmi: {mask_ssmi:.4f} \n\n'
                'val_real_predette: r_r:{r_r:d}, im_0:{im_0:d}, im_1:{im_1:d} / {total_valid}'.format(
                    loss_G2_Bibranch=np.mean(logs_to_print['loss_values_valid_G2_Bibranch']),
                    loss_D=np.mean(logs_to_print['loss_values_valid_D']),
                    loss_D_fake=np.mean(logs_to_print['loss_values_valid_fake_D']),
                    loss_D_real=np.mean(logs_to_print['loss_values_valid_real_D']),
                    r_r=np.sum(logs_to_print['r_r_valid']),
                    im_0=np.sum(logs_to_print['img_0_valid']),
                    im_1=np.sum(logs_to_print['img_1_valid']),
                    ssmi=np.mean(logs_to_print['ssim_valid']),
                    mask_ssmi=np.mean(logs_to_print['mask_ssim_valid']),
                    total_valid=self.config.dataset_valid_len))
            sys.stdout.flush()
            sys.stdout.write('\n\n')

            # -CallBacks
            # --Save weights
            # G2_Bibranch
            name_model = "Model_G2_Bibranch_epoch_{epoch:03d}-" \
                         "loss_{loss:.2f}-" \
                         "ssmi_{ssmi:.2f}-" \
                         "mask_ssmi_{mask_ssim:.2f}-" \
                         "r_r_{r_r:d}-" \
                         "im_0_{im_0:d}-" \
                         "im_1_{im_1:d}-" \
                         "val_loss_{val_loss:.2f}-" \
                         "val_ssim_{val_ssim:.2f}-" \
                         "val_mask_ssim_{val_mask_ssim:.2f}-" \
                         "val_r_r_{val_r_r:d}-" \
                         "val_im_0_{val_im_0:d}-" \
                         "val_im_1_{val_im_1:d}.hdf5".format(
                epoch=epoch + 1,
                loss=np.mean(logs_to_print['loss_values_train_G2_Bibranch']),
                ssmi=np.mean(logs_to_print['ssim_train']),
                mask_ssim=np.mean(logs_to_print['mask_ssim_train']),
                r_r=int(np.sum(logs_to_print['r_r_train'])),
                im_0=int(np.sum(logs_to_print['img_0_train'])),
                im_1=int(np.sum(logs_to_print['img_1_train'])),
                val_loss=np.mean(logs_to_print['loss_values_valid_G2_Bibranch']),
                val_ssim=np.mean(logs_to_print['ssim_valid']),
                val_mask_ssim=np.mean(logs_to_print['mask_ssim_valid']),
                val_r_r=int(np.sum(logs_to_print['r_r_valid'])),
                val_im_0=int(np.sum(logs_to_print['img_0_valid'])),
                val_im_1=int(np.sum(logs_to_print['img_1_valid'])),
            )
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_G2_Bibranch.save_weights(filepath)

            # D
            name_model = "Model_D_epoch_{epoch:03d}-" \
                         "loss_{loss:.2f}-" \
                         "loss_values_D_fake_{loss_D_fake:.2f}-" \
                         "loss_values_D_real_{loss_D_real:.2f}-" \
                         "val_loss_{val_loss:.2f}-" \
                         "val_loss_values_D_fake_{val_loss_D_real:.2f}-" \
                         "val_loss_values_D_real_{val_loss_D_fake:.2f}.hdf5".format(
                epoch=epoch + 1,
                loss=np.mean(logs_to_print['loss_values_valid_D']),
                loss_D_fake=np.mean(logs_to_print['loss_values_valid_fake_D']),
                loss_D_real=np.mean(logs_to_print['loss_values_valid_real_D']),
                val_loss=np.mean(logs_to_print['loss_values_valid_D']),
                val_loss_D_real=np.mean(logs_to_print['loss_values_valid_real_D']),
                val_loss_D_fake=np.mean(logs_to_print['loss_values_valid_fake_D']))
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_D.save_weights(filepath)

            # --Save logs
            history_GAN['epoch'] = epoch + 1
            history_GAN['loss_train_G2_Bibranch'][epoch] = np.mean(logs_to_print['loss_values_train_G2_Bibranch'])
            history_GAN['loss_train_D'][epoch] = np.mean(logs_to_print['loss_values_train_D'])
            history_GAN['loss_train_fake_D'][epoch] = np.mean(logs_to_print['loss_values_train_fake_D'])
            history_GAN['loss_train_real_D'][epoch] = np.mean(logs_to_print['loss_values_train_real_D'])
            history_GAN['ssim_train'][epoch] = np.mean(logs_to_print['ssim_train'])
            history_GAN['mask_ssim_train'][epoch] = np.mean(logs_to_print['mask_ssim_train'])
            history_GAN['r_r_train'][epoch] = np.sum(logs_to_print['r_r_train'])
            history_GAN['img_0_train'][epoch] = np.sum(logs_to_print['img_0_train'])
            history_GAN['img_1_train'][epoch] = np.sum(logs_to_print['img_1_train'])
            history_GAN['loss_valid_G2_Bibranch'][epoch] = np.mean(logs_to_print['loss_values_valid_G2_Bibranch'])
            history_GAN['loss_valid_D'][epoch] = np.mean(logs_to_print['loss_values_valid_D'])
            history_GAN['loss_valid_fake_D'][epoch] = np.mean(logs_to_print['loss_values_valid_fake_D'])
            history_GAN['loss_valid_real_D'][epoch] = np.mean(logs_to_print['loss_values_valid_real_D'])
            history_GAN['ssim_valid'][epoch] = np.mean(logs_to_print['ssim_valid'])
            history_GAN['mask_ssim_valid'][epoch] = np.mean(logs_to_print['mask_ssim_valid'])
            history_GAN['r_r_valid'][epoch] = np.sum(logs_to_print['r_r_valid'])
            history_GAN['img_0_valid'][epoch] = np.sum(logs_to_print['img_0_valid'])
            history_GAN['img_1_valid'][epoch] = np.sum(logs_to_print['img_1_valid'])
            np.save(os.path.join(self.config.logs_path, 'history_GAN.npy'), history_GAN)

            # --Update learning rate
            if epoch % self.config.lr_update_epoch_GAN == self.config.lr_update_epoch_GAN - 1:
                self.opt_G2_Bibranch.lr = self.opt_G2_Bibranch.lr * 0.5
                self.opt_D.lr = self.opt_D.lr * 0.5
                print("-Aggiornamento Learning rate G2_Bibranch: ", self.opt_G2_Bibranch.lr.numpy())
                print("-Aggiornamento Learning rate D: ", self.opt_D.lr.numpy())
                print("\n")

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
        output_G1 = self.model_G1_Bibranch(input_G1)  # [batch, 96, 128, 1] dtype=float32
        output_G1 = tf.cast(output_G1, dtype=tf.float16)

        noise = (np.random.normal(0, 1, image_raw_1.shape) * 0.0010) * tf.math.reduce_sum((pose_1 + 1) /2, axis=-1).numpy().reshape(image_raw_1.shape)
        output_G1 = tf.add(output_G1, noise)

        with tf.GradientTape() as g2_tape:

            # G2_Bibranch
            input_G2_Bibranch = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2_Bibranch = self.model_G2_Bibranch(input_G2_Bibranch)  # [batch, 96, 128, 1] dtype=float32

            # Predizione D
            output_G2_Bibranch = tf.cast(output_G2_Bibranch, dtype=tf.float16)
            refined_result = output_G1 + output_G2_Bibranch  # [batch, 96, 128, 1]
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                                axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = self.model_D(input_D)  # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1])  # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

            # Loss G2_Bibranch
            loss_value_G2_Bibranch = G2_Bibranch.Loss(D_neg_refined_result, refined_result, image_raw_1, mask_1)

        if (id_batch + 1) % 3 == 0:
            print("G2_Bibranch")
            # backprop G2_Bibranch
            self.opt_G2_Bibranch.minimize(loss_value_G2_Bibranch, var_list=self.model_G2_Bibranch.trainable_weights, tape=g2_tape)

        with tf.GradientTape() as d_tape:

            # Predizione G2_Bibranch
            input_G2_Bibranch = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
            output_G2_Bibranch = self.model_G2_Bibranch(input_G2_Bibranch)  # [batch, 96, 128, 1]

            # D
            output_G2_Bibranch = tf.cast(output_G2_Bibranch, dtype=tf.float16)
            refined_result = output_G1 + output_G2_Bibranch  # [batch, 96, 128, 1]
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                                axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
            output_D = self.model_D(input_D)  # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1])  # [batch*3]
            output_D = tf.cast(output_D, dtype=tf.float16)
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

            # Loss D
            loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_pos_image_raw_1, D_neg_refined_result,
                                                                    D_neg_image_raw_0)

        if not (id_batch + 1) % 3 == 0:
            # backprop D
            self.opt_D.minimize(loss_value_D, var_list=self.model_D.trainable_weights, tape=d_tape)


        # Metrics
        # - SSIM
        ssim_value = G2_Bibranch.m_ssim(refined_result, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G2_Bibranch.mask_ssim(refined_result, image_raw_1, mask_1, mean_0, mean_1)

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
            name_directory = os.path.join("./results_ssim/GAN/train", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G2_Bibranch_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
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
            file = open(name_directory + '/' + 'G2_Bibranch_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                           batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G2_Bibranch.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
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
        output_G1 = self.model_G1_Bibranch(input_G1)  # output_g1 --> [batch, 96, 128, 1]

        # G2_Bibranch
        output_G1 = tf.cast(output_G1, dtype=tf.float16)
        input_G2_Bibranch = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 96, 128, 2]
        output_G2_Bibranch = self.model_G2_Bibranch(input_G2_Bibranch)  # [batch, 96, 128, 1]

        # D
        output_G2_Bibranch = tf.cast(output_G2_Bibranch, dtype=tf.float16)
        refined_result = output_G1 + output_G2_Bibranch  # [batch, 96, 128, 1]
        input_D = tf.concat([image_raw_1, refined_result, image_raw_0],
                            axis=0)  # [batch * 3, 96, 128, 1] --> batch * 3 poichè concateniamo sul primo asse
        output_D = self.model_D(input_D)  # [batch * 3, 1]
        output_D = tf.reshape(output_D, [-1])  # [batch*3]
        output_D = tf.cast(output_D, dtype=tf.float16)
        D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

        # Loss
        loss_value_G2_Bibranch = G2_Bibranch.Loss(D_neg_refined_result, refined_result, image_raw_1, mask_1)
        loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_pos_image_raw_1, D_neg_refined_result,
                                                                D_neg_image_raw_0)

        # Metrics
        # - SSIM
        ssim_value = G2_Bibranch.m_ssim(refined_result, image_raw_1, mean_0, mean_1)
        mask_ssim_value = G2_Bibranch.mask_ssim(refined_result, image_raw_1, mask_1, mean_0, mean_1)

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
            name_directory = os.path.join("./results_ssim/GAN/valid", str(epoch + 1))
            if not os.path.exists(name_directory):
                os.mkdir(name_directory)
            name_grid = os.path.join(name_directory,
                                     'G2_Bibranch_epoch_{epoch}_batch_{batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
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
            file = open(name_directory + '/' + 'G2_Bibranch_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                           batch=id_batch), "w")
            file.write(txt_file)
            file.close()

        return loss_value_G2_Bibranch.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               real_predette_refined_result_train.shape[0], real_predette_image_raw_0_train.shape[0], \
               real_predette_image_raw_1_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy()


if __name__ == "__main__":
    Config_file = __import__('B1_config_utils')
    config = Config_file.Config()
    config.print_info()
    config.save_info()

    pg2 = PG2(config)  # Pose Guided ^2 network

    if config.trainig_G1_Bibranch:
        pg2.train_G1_Bibranch()
    elif config.trainig_GAN:
        pg2.train_conditional_GAN()