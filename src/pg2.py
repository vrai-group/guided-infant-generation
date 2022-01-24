"""
Questo script consente di avviare il training del G1 e della GAN
"""
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils

from utils.evaluation.tsne import vgg16_pca_tsne_features_real

class PG2(object):

    def __init__(self, config):
        self.config = config

        # -Import dinamico dell modulo di preprocess dataset Ad esempio: Syntetich
        name_module_preprocess_dataset = config.DATASET.split('_')[0]
        self.dataset_module = utils.import_module(name_module_preprocess_dataset, config.dataset_module_dir_path)

        # -Import dinamico dell'architettura
        self.G1 = utils.import_module(name_module="G1", path=config.models_dir_path).G1()
        self.G2 = utils.import_module(name_module="G2", path=config.models_dir_path).G2()
        self.D = utils.import_module(name_module="D", path=config.models_dir_path).D()

    def _save_grid(self, epoch, id_batch, batch, output, ssim_value, mask_ssim_value, type, architecture):
        pz_0 = batch[5]  # [batch, 1]
        pz_1 = batch[6]  # [batch, 1]
        name_0 = batch[7]  # [batch, 1]
        name_1 = batch[8]  # [batch, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))

        # GRID: Save griglia di immagini predette
        name_directory = os.path.join(self.config.grid_dir_path, architecture, type, str(epoch + 1))
        if not os.path.exists(name_directory):
            os.mkdir(name_directory)
        name_grid = os.path.join(name_directory,
                                 architecture + '_epoch_{epoch}_batch_{id_batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                     epoch=epoch + 1,
                                     id_batch=id_batch,
                                     ssim=ssim_value,
                                     mask_ssim=mask_ssim_value))
        mean_0 = tf.cast(mean_0, dtype=tf.float32)
        output = self.dataset_module.unprocess_image(output, mean_0, 32765.5)
        utils.save_grid(output, name_grid)  # si salva in una immagine contenente una griglia tutti i  G1 + DiffMap

        stack_pairs = np.c_[pz_0.numpy(), name_0.numpy(), pz_1.numpy(), name_1.numpy()]
        stack_pairs = np.array(
            [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
             stack_pairs])
        txt_file = 'pz_pair: \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
        file = open(name_directory + '/' + architecture + '_epoch_{epoch}_batch_{batch}.txt'.format(epoch=epoch + 1,
                                                                                                    batch=id_batch),
                    "w")
        file.write(txt_file)
        file.close()

    def train_G1(self):

        # - LOGS
        path_history_G1 = os.path.join(self.config.logs_dir_path, 'history_G1.npy')
        history_G1 = {'epoch': 0,
                      'loss_train': np.empty((self.config.G1_epochs)),
                      'ssim_train': np.empty((self.config.G1_epochs)),
                      'mask_ssim_train': np.empty((self.config.G1_epochs)),

                      'loss_valid': np.empty((self.config.G1_epochs)),
                      'ssim_valid': np.empty((self.config.G1_epochs)),
                      'mask_ssim_valid': np.empty((self.config.G1_epochs))}

        # Se esistenti, precarico i logs
        if os.path.exists(path_history_G1):
            old_history_G1 = np.load(path_history_G1, allow_pickle='TRUE')
            epoch = old_history_G1[()]['epoch']
            for key, value in old_history_G1.item().items():
                if key == 'epoch':
                    history_G1[key] = value
                else:
                    history_G1[key][:epoch] = value[:epoch]

        # - DATASET: caricamento
        dataset_train = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_train)
        dataset_train = dataset_train.batch(1)

        dataset_valid = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_train)
        dataset_valid = dataset_valid.batch(1)

        # TRAIN: epoch
        for epoch in range(history_G1['epoch'], self.config.G1_epochs):
            train_iterator = iter(dataset_train)
            valid_iterator = iter(dataset_valid)

            # -DATASET: augumentazione
            name_tfrecord_aug_train, dataset_train_aug_len = utils.apply_augumentation(data_tfrecord_path=self.config.data_tfrecord_path,
                                                             unprocess_dataset_iterator=train_iterator,
                                                             name_dataset="train",
                                                             len_dataset=self.config.dataset_train_len)
            name_tfrecord_aug_valid, dataset_valid_aug_len = utils.apply_augumentation(data_tfrecord_path=self.config.data_tfrecord_path,
                                                             unprocess_dataset_iterator=valid_iterator,
                                                             name_dataset="valid",
                                                             len_dataset=self.config.dataset_valid_len)

            print("\nAugumentazione terminata: ")
            print("- lunghezza train: ", dataset_train_aug_len)
            print("- lunghezza valid: ", dataset_valid_aug_len)
            print("\n")

            # DATASET augumentato: pipeline preprocessamento
            dataset_train_aug_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=name_tfrecord_aug_train)
            dataset_train_aug = self.dataset_module.preprocess_dataset(dataset_train_aug_unp)
            dataset_train_aug = dataset_train_aug.shuffle(dataset_train_aug_len, reshuffle_each_iteration=True)
            dataset_train_aug = dataset_train_aug.batch(self.config.G1_batch_size_train)
            dataset_train_aug = dataset_train_aug.prefetch(tf.data.AUTOTUNE)

            dataset_valid_aug_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=name_tfrecord_aug_train)
            dataset_valid_aug = self.dataset_module.preprocess_dataset(dataset_valid_aug_unp)
            dataset_valid_aug = dataset_valid_aug.shuffle(dataset_valid_aug_len, reshuffle_each_iteration=True)
            dataset_valid_aug = dataset_valid_aug.batch(self.config.G1_batch_size_valid)
            dataset_valid_aug = dataset_valid_aug.prefetch(tf.data.AUTOTUNE)

            # calcolo l'effettivo numero di bacthes considerando la grandezza del dataset di augumentazione
            num_batches_train = dataset_train_aug_len // self.config.G1_batch_size_train
            num_batches_valid = dataset_valid_aug_len // self.config.G1_batch_size_valid

            # rinizializzo gli iteratori sul dataset augumentato
            train_aug_iterator = iter(dataset_train_aug)
            valid_aug_iterator = iter(dataset_valid_aug)

            # dizionario utilizzato per salvare i valori per ogni epoca in modo tale da printare a schermo le medie di step
            # in step. Lo rinizializzo ad ogni nuova epoca
            logs_to_print = {'loss_values_train': np.empty((num_batches_train)),
                             'ssim_train': np.empty((num_batches_train)),
                             'mask_ssim_train': np.empty((num_batches_train)),

                             'loss_values_valid': np.empty((num_batches_valid)),
                             'ssim_valid': np.empty((num_batches_valid)),
                             'mask_ssim_valid': np.empty((num_batches_valid))
                             }

            # TRAIN: iteration
            for id_batch in range(num_batches_train):
                batch = next(train_aug_iterator)
                logs_to_print['loss_values_train'][id_batch], logs_to_print['ssim_train'][id_batch], \
                logs_to_print['mask_ssim_train'][id_batch], I_PT1 = self._train_on_batch_G1(batch)

                # Grid
                if epoch % self.config.G1_save_grid_ssim_epoch_train == self.config.G1_save_grid_ssim_epoch_train - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT1, logs_to_print['ssim_train'][id_batch],
                                    logs_to_print['mask_ssim_train'][id_batch], type="valid", architecture="G1")

                # Logs a schermo
                sys.stdout.write('\rEpoch {epoch} step {id_batch} / {num_batches} --> \
                                  loss_G1: {loss_G1:.4f}, ssmi: {ssmi:.4f}, mask_ssmi: {mask_ssmi:.4f}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch + 1,
                    num_batches=num_batches_train,
                    loss_G1=np.mean(logs_to_print['loss_values_train'][:id_batch + 1]),
                    ssmi=np.mean(logs_to_print['ssim_train'][:id_batch + 1]),
                    mask_ssmi=np.mean(logs_to_print['mask_ssim_train'][:id_batch + 1])))
                sys.stdout.flush()

            sys.stdout.write('\nValidazione..\n')
            sys.stdout.flush()

            # VALID: iteration
            for id_batch in range(num_batches_valid):
                batch = next(valid_aug_iterator)
                logs_to_print['loss_values_valid'][id_batch], logs_to_print['ssim_valid'][id_batch], \
                logs_to_print['mask_ssim_valid'][id_batch], I_PT1 = self._valid_on_batch_G1(batch)

                if epoch % self.config.save_grid_ssim_epoch_valid == self.config.save_grid_ssim_epoch_valid - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT1, logs_to_print['ssim_valid'][id_batch],
                                    logs_to_print['mask_ssim_valid'][id_batch], type="valid", architecture="G1")

                sys.stdout.write('\r{id_batch} / {total}'.format(id_batch=id_batch + 1, total=num_batches_valid))
                sys.stdout.flush()

            sys.stdout.write('\r\rval_loss_G1: {loss_G1:.4f}, val_ssmi: {ssmi:.4f}, val_mask_ssmi: {mask_ssmi:.4f}'.format(
                loss_G1=np.mean(logs_to_print['loss_values_valid']),
                ssmi=np.mean(logs_to_print['ssim_valid']),
                mask_ssmi=np.mean(logs_to_print['mask_ssim_valid'])))
            sys.stdout.flush()
            sys.stdout.write('\n\n')

            # -CallBacks
            # --Save weights
            name_model = 'Model_G1_epoch_{epoch:03d}-' \
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
            filepath = os.path.join(self.config.G1_weigths_path, name_model)
            self.G1.save_weights(filepath)

            # --Update learning rate
            if epoch % self.config.G1_lr_update_epoch == self.config.G1_lr_update_epoch - 1:
                self.G1.opt.lr = self.G1.opt.lr * self.config.G1_drop_rate
                print("-Aggiornamento Learning rate G1: ", self.G1.opt.lr.numpy())
                print("\n")

            # --Save logs
            history_G1['epoch'] = epoch + 1
            history_G1['loss_train'][epoch] = np.mean(logs_to_print['loss_values_train'])
            history_G1['ssim_train'][epoch] = np.mean(logs_to_print['ssim_train'])
            history_G1['mask_ssim_train'][epoch] = np.mean(logs_to_print['mask_ssim_train'])
            history_G1['loss_valid'][epoch] = np.mean(logs_to_print['loss_values_valid'])
            history_G1['ssim_valid'][epoch] = np.mean(logs_to_print['ssim_valid'])
            history_G1['mask_ssim_valid'][epoch] = np.mean(logs_to_print['mask_ssim_valid'])
            np.save(path_history_G1, history_G1)

        print("#############\n\n")

    def _train_on_batch_G1(self, batch):
        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        with tf.GradientTape() as g1_tape:
            I_PT1 = self.G1.prediction(Ic, Pt)
            loss_value_G1 = self.G1.PoseMaskloss(I_PT1, It, Mt)
        self.G1.opt.minimize(loss_value_G1, var_list=self.G1.model.trainable_weights, tape=g1_tape)

        # METRICS
        ssim_value = self.G1.ssim(I_PT1, It, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)
        mask_ssim_value = self.G1.mask_ssim(I_PT1, It, Mt, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)

        return loss_value_G1, ssim_value, mask_ssim_value, I_PT1

    def _valid_on_batch_G1(self, batch):

        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        I_PT1 = self.G1.prediction(Ic, Pt)
        loss_value_G1 = self.G1.PoseMaskloss(I_PT1, It, Mt)

        # METRICS
        ssim_value = self.G1.ssim(I_PT1, It, mean_0, mean_1)
        mask_ssim_value = self.G1.mask_ssim(I_PT1, It, Mt, mean_0, mean_1)

        return loss_value_G1, ssim_value, mask_ssim_value, I_PT1

    def train_cDCGAN(self):
        # Note: G1 è preaddestrato

        # -History del training
        path_history_GAN = os.path.join(self.config.logs_dir_path, 'history_GAN.npy')
        history_GAN = {'epoch': 0,
                       'loss_train_G2': np.empty((self.config.GAN_epochs)),
                       'loss_train_D': np.empty((self.config.GAN_epochs)),
                       'loss_train_fake_D': np.empty((self.config.GAN_epochs)),
                       'loss_train_real_D': np.empty((self.config.GAN_epochs)),
                       'ssim_train': np.empty((self.config.GAN_epochs)),
                       'mask_ssim_train': np.empty((self.config.GAN_epochs)),
                       'r_r_train': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'img_0_train': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'img_1_train': np.empty((self.config.GAN_epochs), dtype=np.uint32),

                       'loss_valid_G2': np.empty((self.config.GAN_epochs)),
                       'loss_valid_D': np.empty((self.config.GAN_epochs)),
                       'loss_valid_fake_D': np.empty((self.config.GAN_epochs)),
                       'loss_valid_real_D': np.empty((self.config.GAN_epochs)),
                       'ssim_valid': np.empty((self.config.GAN_epochs)),
                       'mask_ssim_valid': np.empty((self.config.GAN_epochs)),
                       'r_r_valid': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'img_0_valid': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'img_1_valid': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       }

        # Se esistenti, precarico i logs
        if os.path.exists(os.path.join(path_history_GAN, 'history_GAN.npy')):
            old_history_GAN = np.load(os.path.join(self.config.logs_path, 'history_GAN.npy'), allow_pickle='TRUE')
            # epoch = old_history_G1[()]['epoch'] --> anche in questo modi riesco ad ottenere il value dell'epoca
            epoch = old_history_GAN.item().get('epoch')
            for key, value in old_history_GAN.item().items():
                if key == 'epoch':
                    history_GAN[key] = value
                else:
                    history_GAN[key][:epoch] = value[:epoch]

        # DATASET: Caricamento dataset
        dataset_train_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_train)
        dataset_train = self.dataset_module.preprocess_dataset(dataset_train_unp)
        dataset_train = dataset_train.shuffle(dataset_train, reshuffle_each_iteration=True)
        dataset_train = dataset_train.batch(self.config.G1_batch_size_train)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

        dataset_valid_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_valid)
        dataset_valid = self.dataset_module.preprocess_dataset(dataset_valid_unp)
        dataset_valid = dataset_valid.batch(self.config.G1_batch_size_valid)
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)

        num_batches_train = self.config.dataset_train_len // self.config.GAN_batch_size_train
        num_batches_valid = self.config.dataset_valid_len // self.config.GAN_batch_size_valid

        # Carico il modello preaddestrato G1
        self.G1.model.load_weights(self.config.G1_weigths_file_path)
        #self.model_G1.summary()

        # TRAIN: epoch
        for epoch in range(history_GAN['epoch'], self.config.GAN_epochs):
            train_iterator = iter(dataset_train)
            valid_iterator = iter(dataset_valid)

            # Vettori che mi serviranno per salvare i valori per ogni epoca in modo tale da printare a schermo le medie
            logs_to_print = {'loss_values_train_G2': np.empty((num_batches_train)),
                             'loss_values_train_D': np.empty((num_batches_train)),
                             'loss_values_train_fake_D': np.empty((num_batches_train)),
                             'loss_values_train_real_D': np.empty((num_batches_train)),
                             'ssim_train': np.empty((num_batches_train)),
                             'mask_ssim_train': np.empty((num_batches_train)),
                             'r_r_train': np.empty((num_batches_train), dtype=np.uint32),
                             'img_0_train': np.empty((num_batches_train), dtype=np.uint32),
                             'img_1_train': np.empty((num_batches_train), dtype=np.uint32),

                             'loss_values_valid_G2': np.empty((num_batches_valid)),
                             'loss_values_valid_D': np.empty((num_batches_valid)),
                             'loss_values_valid_fake_D': np.empty((num_batches_valid)),
                             'loss_values_valid_real_D': np.empty((num_batches_valid)),
                             'ssim_valid': np.empty((num_batches_valid)),
                             'mask_ssim_valid': np.empty((num_batches_valid)),
                             'r_r_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             'img_0_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             'img_1_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             }

            # TRAIN: iteration
            for id_batch in range(num_batches_train):
                batch = next(train_iterator)
                logs_to_print['loss_values_train_G2'][id_batch], logs_to_print['loss_values_train_D'][id_batch], \
                logs_to_print['loss_values_train_fake_D'][id_batch], logs_to_print['loss_values_train_real_D'][id_batch], \
                logs_to_print['r_r_train'][id_batch], logs_to_print['img_0_train'][id_batch], \
                logs_to_print['img_1_train'][id_batch], logs_to_print['ssim_train'][id_batch], \
                logs_to_print['mask_ssim_train'][id_batch], I_PT2 = \
                    self._train_on_batch_cDCGAN(id_batch, batch)

                # GRID
                if epoch % self.config.GAN_save_grid_ssim_epoch_train == self.config.GAN_save_grid_ssim_epoch_train.save_grid_ssim_epoch_train - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT2, logs_to_print['ssim_train'][id_batch],
                                    logs_to_print['mask_ssim_train'][id_batch], type="train", architecture="GAN")
                # Logs a schermo
                sys.stdout.write('\rEpoch {epoch} step {id_batch} / {num_batches} --> loss_G2: {loss_G2:2f}, '
                                 'loss_D: {loss_D:2f}, loss_D_fake: {loss_D_fake:2f}, loss_D_real: {loss_D_real:2f}, '
                                 'ssmi: {ssmi:2f}, mask_ssmi: {mask_ssmi:2f}, real_predette:: r_r:{r_r:1}, '
                                 'im_0:{im_0:1}, im_1:{im_1:1} / {total_train}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch + 1,
                    num_batches=num_batches_train,
                    loss_G2=np.mean(logs_to_print['loss_values_train_G2'][:id_batch + 1]),
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

            sys.stdout.write('\nValidazione..\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                batch = next(valid_iterator)
                logs_to_print['loss_values_valid_G2'][id_batch], logs_to_print['loss_values_valid_D'][id_batch], \
                logs_to_print['loss_values_valid_fake_D'][id_batch], logs_to_print['loss_values_valid_real_D'][id_batch], \
                logs_to_print['r_r_valid'][id_batch], logs_to_print['img_0_valid'][id_batch], \
                logs_to_print['img_1_valid'][id_batch], logs_to_print['ssim_valid'][id_batch], \
                logs_to_print['mask_ssim_valid'][id_batch], I_PT2 = self._valid_on_batch_cDCGAN(batch)

                sys.stdout.write('\r{id_batch} / {total}'.format(id_batch=id_batch + 1, total=num_batches_valid))
                sys.stdout.flush()

                if epoch % self.config.GAN_save_grid_ssim_epoch_valid == self.config.GAN_save_grid_ssim_epoch_valid - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT2, logs_to_print['ssim_valid'][id_batch],
                                    logs_to_print['mask_ssim_valid'][id_batch], type="valid", architecture="GAN")

            sys.stdout.write('')
            sys.stdout.write('\r\r'
                'val_loss_G2: {loss_G2:.4f}, val_loss_D: {loss_D:.4f}, val_loss_D_fake: {loss_D_fake:.4f}, '
                'val_loss_D_real: {loss_D_real:.4f}, val_ssmi: {ssmi:.4f}, val_mask_ssmi: {mask_ssmi:.4f} \n\n'
                'val_real_predette: r_r:{r_r:d}, im_0:{im_0:d}, im_1:{im_1:d} / {total_valid}'.format(
                    loss_G2=np.mean(logs_to_print['loss_values_valid_G2']),
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

            # -CallBacks
            # --Save weights
            # G2
            name_model = "Model_G2_epoch_{epoch:03d}-" \
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
                loss=np.mean(logs_to_print['loss_values_train_G2']),
                ssmi=np.mean(logs_to_print['ssim_train']),
                mask_ssim=np.mean(logs_to_print['mask_ssim_train']),
                r_r=int(np.sum(logs_to_print['r_r_train'])),
                im_0=int(np.sum(logs_to_print['img_0_train'])),
                im_1=int(np.sum(logs_to_print['img_1_train'])),
                val_loss=np.mean(logs_to_print['loss_values_valid_G2']),
                val_ssim=np.mean(logs_to_print['ssim_valid']),
                val_mask_ssim=np.mean(logs_to_print['mask_ssim_valid']),
                val_r_r=int(np.sum(logs_to_print['r_r_valid'])),
                val_im_0=int(np.sum(logs_to_print['img_0_valid'])),
                val_im_1=int(np.sum(logs_to_print['img_1_valid'])),
            )
            filepath = os.path.join(self.config.GAN_weigths_path, name_model)
            self.G2.models.save_weights(filepath)

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
            filepath = os.path.join(self.config.GAN_weigths_path, name_model)
            self.D.model.save_weights(filepath)

            # --Save logs
            history_GAN['epoch'] = epoch + 1
            history_GAN['loss_train_G2'][epoch] = np.mean(logs_to_print['loss_values_train_G2'])
            history_GAN['loss_train_D'][epoch] = np.mean(logs_to_print['loss_values_train_D'])
            history_GAN['loss_train_fake_D'][epoch] = np.mean(logs_to_print['loss_values_train_fake_D'])
            history_GAN['loss_train_real_D'][epoch] = np.mean(logs_to_print['loss_values_train_real_D'])
            history_GAN['ssim_train'][epoch] = np.mean(logs_to_print['ssim_train'])
            history_GAN['mask_ssim_train'][epoch] = np.mean(logs_to_print['mask_ssim_train'])
            history_GAN['r_r_train'][epoch] = np.sum(logs_to_print['r_r_train'])
            history_GAN['img_0_train'][epoch] = np.sum(logs_to_print['img_0_train'])
            history_GAN['img_1_train'][epoch] = np.sum(logs_to_print['img_1_train'])
            history_GAN['loss_valid_G2'][epoch] = np.mean(logs_to_print['loss_values_valid_G2'])
            history_GAN['loss_valid_D'][epoch] = np.mean(logs_to_print['loss_values_valid_D'])
            history_GAN['loss_valid_fake_D'][epoch] = np.mean(logs_to_print['loss_values_valid_fake_D'])
            history_GAN['loss_valid_real_D'][epoch] = np.mean(logs_to_print['loss_values_valid_real_D'])
            history_GAN['ssim_valid'][epoch] = np.mean(logs_to_print['ssim_valid'])
            history_GAN['mask_ssim_valid'][epoch] = np.mean(logs_to_print['mask_ssim_valid'])
            history_GAN['r_r_valid'][epoch] = np.sum(logs_to_print['r_r_valid'])
            history_GAN['img_0_valid'][epoch] = np.sum(logs_to_print['img_0_valid'])
            history_GAN['img_1_valid'][epoch] = np.sum(logs_to_print['img_1_valid'])
            np.save(os.path.join(path_history_GAN, 'history_GAN.npy'), history_GAN)

            # --Update learning rate
            if epoch % self.config.GAN_lr_update_epoch == self.config.GAN_lr_update_epoch - 1:
                self.G2.optimizer.lr = self.G2.optimizer.lr * self.config.GAN_G2_drop_rate
                self.D.optimizer.lr = self.D.optimizer.lr * self.config.GAN_D_drop_rate
                print("-Aggiornamento Learning rate G2: ", self.G2.optimizer.lr.numpy())
                print("-Aggiornamento Learning rate D: ", self.D.optimizer.lr.numpy())
                print("\n")

            print("#######")

    def _train_on_batch_cDCGAN(self, id_batch, batch):

        def _tape(loss_function_G2, loss_function_D):
            with tf.GradientTape() as tape:
                I_D = self.G2.prediction(I_PT1, Ic)
                I_D = tf.cast(I_D, dtype=tf.float16)
                I_PT2 = I_PT1 + I_D  # [batch, 96, 128, 1]

                output_D = self.D.prediction(It, I_PT2, Ic)
                output_D = tf.cast(output_D, dtype=tf.float16)
                D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

                loss_value_G2 = loss_function_G2(D_neg_refined_result, I_PT2, It, Mt)
                loss_value_D = loss_function_D(D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0)

            return tape, loss_value_G2, loss_value_D, I_PT2, I_D, D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0


        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        # G1
        I_PT1 = self.G1.prediction(Ic, Pt)
        I_PT1 = tf.cast(I_PT1, dtype=tf.float16)

        # BACKPROP G2
        I_PT2 = None
        D_neg_refined_result, D_neg_image_raw_0, D_pos_image_raw_1 = None, None, None
        loss_value_G2, loss_value_D, loss_fake, loss_real = None, None, None, None
        if (id_batch + 1) % 3 == 0:
            g2_tape, loss_value_G2, loss_value_D, I_PT2, I_D, \
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = _tape(self.G2.adv_loss, self.D.adv_loss)
            loss_value_D, loss_fake, loss_real = loss_value_D
            self.G2.optimizer.minimize(loss_value_G2, var_list=self.G2.optimizer.trainable_weights, tape=g2_tape)

        # BACKPROP D
        if not (id_batch + 1) % 3 == 0:
            d_tape, loss_value_G2, loss_value_D, I_PT2, I_D, \
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = _tape(self.G2.adv_loss, self.D.adv_loss)
            loss_value_D, loss_fake, loss_real = loss_value_D
            self.D.optimizer.minimize(loss_value_D, var_list=self.D.optimizer.trainable_weights, tape=d_tape)

        # Metrics
        # - SSIM
        ssim_value = self.G2.ssim(I_PT2, It, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)
        mask_ssim_value = self.G2.mask_ssim(I_PT2, It, Mt, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)

        # - Real predette di refined_result dal discriminatore
        np_array_D_neg_refined_result = D_neg_refined_result.numpy()
        real_predette_refined_result_train = np_array_D_neg_refined_result[np_array_D_neg_refined_result > 0]

        # - Real predette di image_raw_0 dal discriminatore
        np_array_D_neg_image_raw_0 = D_neg_image_raw_0.numpy()
        real_predette_image_raw_0_train = np_array_D_neg_image_raw_0[np_array_D_neg_image_raw_0 > 0]

        # - Real predette di image_raw_1 (Target) dal discriminatore
        np_array_D_pos_image_raw_1 = D_pos_image_raw_1.numpy()
        real_predette_image_raw_1_train = np_array_D_pos_image_raw_1[np_array_D_pos_image_raw_1 > 0]

        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               real_predette_refined_result_train.shape[0], real_predette_image_raw_0_train.shape[0], \
               real_predette_image_raw_1_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy(), I_PT2

    def _valid_on_batch_cDCGAN(self, batch):
        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        # G1
        I_PT1 = self.G1.prediction(Ic, Pt)
        I_PT1 = tf.cast(I_PT1, dtype=tf.float16)

        # G2
        I_D = self.G2.prediction(I_PT1, Ic)
        I_D = tf.cast(I_D, dtype=tf.float16)
        I_PT2 = I_PT1 + I_D  # [batch, 96, 128, 1]

        # D
        output_D = self.D.prediction(It, I_PT2, Ic)
        output_D = tf.cast(output_D, dtype=tf.float16)
        D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D, 3)  # [batch]

        # Loss
        loss_value_G2 = self.G2.adv_loss(D_neg_refined_result, I_PT2, It, Ic, Mt)
        loss_value_D, loss_fake, loss_real = self.D.Loss(D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0)

        # Metrics
        # - SSIM
        ssim_value = self.G2.m_ssim(I_PT2, It, mean_0, mean_1)
        mask_ssim_value = self.G2.mask_ssim(I_PT2, It, Mt, mean_0, mean_1)

        # - Real predette di refined_result dal discriminatore
        np_array_D_neg_refined_result = D_neg_refined_result.numpy()
        real_predette_refined_result_train = np_array_D_neg_refined_result[np_array_D_neg_refined_result > 0]

        # - Real predette di image_raw_0 dal discriminatore
        np_array_D_neg_image_raw_0 = D_neg_image_raw_0.numpy()
        real_predette_image_raw_0_train = np_array_D_neg_image_raw_0[np_array_D_neg_image_raw_0 > 0]

        # - Real predette di image_raw_1 (Target) dal discriminatore
        np_array_D_pos_image_raw_1 = D_pos_image_raw_1.numpy()
        real_predette_image_raw_1_train = np_array_D_pos_image_raw_1[np_array_D_pos_image_raw_1 > 0]

        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               real_predette_refined_result_train.shape[0], real_predette_image_raw_0_train.shape[0], \
               real_predette_image_raw_1_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy(), I_PT2

    def inference_on_test(self):
        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_test)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)
        dataset_iterator = iter(dataset)

        # Model
        self.G1.model.load_weights(self.config.G1_weigths_file_path)
        self.G2.model.load_weights(self.config.G2_weigths_file_path)

        for i in range(self.config.dataset_test_len):
            sys.stdout.write("\rProcessamento immagine {cnt} / {tot}".format(cnt=i + 1, tot=self.config.dataset_test_len))
            sys.stdout.flush()
            batch = next(dataset_iterator)
            Ic = batch[0]  # [batch, 96, 128, 1]
            It = batch[1]  # [batch, 96,128, 1]
            Pt = batch[2]  # [batch, 96,128, 14]
            Mt = batch[3]  # [batch, 96,128, 1]
            mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
            mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

            # Predizione
            I_PT1 = self.G1.prediction(Ic, Pt)
            I_D = self.G2.prediction(I_PT1, Ic)
            I_PT2 = I_D + I_PT1

            # Unprocess
            Ic = tf.cast(self.dataset_module.unprocess_image(Ic, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
            It = tf.cast(self.dataset_module.unprocess_image(It, mean_1, 32765.5), dtype=tf.uint16)[0].numpy()

            Mt = tf.cast(Mt, dtype=tf.uint16)[0].numpy().reshape(96, 128, 1)

            I_PT1 = tf.cast(self.dataset_module.unprocess_image(I_PT1, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
            I_PT2 = tf.cast(self.dataset_module.unprocess_image(I_PT2, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()

            # Plot Figure
            fig = plt.figure(figsize=(10, 2))
            columns = 5
            rows = 1
            imgs = [I_PT2, I_PT1, Ic, It, It, Mt]
            labels = ["I_PT2", "I_PT1", "Ic", "It", "Pt", "Mt"]
            for j in range(1, columns * rows + 1):
                sub = fig.add_subplot(rows, columns, j)
                sub.set_title(labels[j - 1])
                plt.imshow(imgs[j - 1], cmap='gray')
            plt.show()

            #Save figure
            # pz_0 = batch[5]  # [batch, 1]
            # pz_1 = batch[6]  # [batch, 1]
            # name_0 = batch[7]  # [batch, 1]
            # name_1 = batch[8]  # [batch, 1]
            # pz_0 = pz_0.numpy()[0].decode("utf-8")
            # pz_1 = pz_1.numpy()[0].decode("utf-8")
            # id_0 = name_0.numpy()[0].decode("utf-8").split('_')[0]  # id dell immagine
            # id_1 = name_1.numpy()[0].decode("utf-8").split('_')[0]
            # name_img = os.path.join(name_dir_to_save_img,
            #                         "{id}-{pz_0}_{id_0}-{pz_1}_{id_1}.png".format(
            #                             id=i,
            #                             pz_0=pz_0,
            #                             pz_1=pz_1,
            #                             id_0=id_0,
            #                             id_1=id_1))
            #
            # plt.savefig(name_img)
            # plt.close(fig)

    # Valutazione metrice IS e FID
    def evaluate_G1(self, analysis="test", bool_save_image=True, batch_size=10):

        path = "evaluation/G1"
        os.makedirs(path, exist_ok=False)

        name_dataset, dataset_len = None, None
        if analysis == "test":
            name_dataset = self.config.name_tfrecord_test
            dataset_len = self.config.dataset_test_len
        elif analysis == "valid":
            name_dataset = self.config.name_tfrecord_valid
            dataset_len = self.config.dataset_valid_len

        # Dataset
        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=name_dataset)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)

        for weight_G1 in os.listdir(self.config.G1_weigths_dir_path):
            num_epoch = weight_G1.split('-')[0].split('_')[3]

            # Directory
            path_evaluation = os.path.join(path, analysis+'_score_epoch'+num_epoch) # directory dove salvare i risultati degli score
            path_imgs = os.path.join(path_evaluation, "imgs")
            path_embeddings = os.path.join(path_evaluation, "inception_embeddings")

            os.makedirs(path_evaluation, exist_ok=True)
            os.makedirs(path_imgs, exist_ok=True)
            os.makedirs(path_embeddings, exist_ok=True)

            # Model
            self.G1.model.load_weights(os.path.join(self.config.G1_weigths_dir_path, weight_G1))

            # Pipiline score
            utils.evaluation_G1.start(self.G1, iter(dataset), dataset_len, batch_size,
                                dataset_module=self.dataset_module, bool_save_img=bool_save_image,
                                path_evaluation=path_evaluation, path_imgs=path_imgs,
                                path_embeddings=path_embeddings)

    def evaluate_GAN(self, analysis="test", bool_save_image=True, batch_size=10):

        # controllo che G1 abbia i pesi settati
        assert self.config.G1_weigths_file_path != None

        path = "evaluation/G2"
        os.makedirs(path, exist_ok=False)

        name_dataset, dataset_len = None, None
        if analysis == "test":
            name_dataset = self.config.name_tfrecord_test
            dataset_len = self.config.dataset_test_len
        elif analysis == "valid":
            name_dataset = self.config.name_tfrecord_valid
            dataset_len = self.config.dataset_valid_len

        # Dataset
        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=name_dataset)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)

        for weight_G2 in os.listdir(self.config.GAN_weigths_dir_path):
            num_epoch = weight_G2.split('-')[0].split('_')[3]

            # Directory
            path_evaluation = os.path.join(path, analysis + '_score_epoch' + num_epoch)  # directory dove salvare i risultati degli score
            path_imgs = os.path.join(path_evaluation, "imgs")
            path_embeddings = os.path.join(path_evaluation, "inception_embeddings")

            os.makedirs(path_evaluation, exist_ok=True)
            os.makedirs(path_imgs, exist_ok=True)
            os.makedirs(path_embeddings, exist_ok=True)

            # Model
            self.G2.model.load_weights(os.path.join(self.config.GAN_weigths_dir_path, weight_G2))

            # Pipiline score
            utils.evaluation_GAN.start(self.G1, self.G2, iter(dataset), dataset_len, batch_size,
                                dataset_module=self.dataset_module, bool_save_img=bool_save_image,
                                path_evaluation=path_evaluation, path_imgs=path_imgs,
                                path_embeddings=path_embeddings)

    def tsne(self):

        list_sets = [[self.config.name_tfrecord_train, self.config.dataset_train_len],
                     [self.config.name_tfrecord_valid, self.config.dataset_valid_len],
                     [self.config.name_tfrecord_test, self.config.dataset_test_len]]
        list_perplexity = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 200, 300]

        # Features real
        vgg16_pca_tsne_features_real.start(list_sets, list_perplexity, self.dataset_module, self.config)


