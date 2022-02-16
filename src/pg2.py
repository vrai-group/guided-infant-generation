import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils

class PG2(object):

    def __init__(self, config):
        self.config = config

        # -Import dinamico dell modulo di preprocess dataset Ad esempio: Syntetich
        self.dataset_module = utils.import_module(path=config.dataset_module_dir_path, name_module=config.DATASET_TYPE)

        # -Import dinamico dell'architettura
        self.G1 = utils.import_module(path=config.models_dir_path, name_module="G1").G1()
        self.G2 = utils.import_module(path=config.models_dir_path, name_module="G2").G2()
        self.D = utils.import_module(path=config.models_dir_path, name_module="D").D()

    def _save_grid(self, epoch, id_batch, batch, output, ssim_value, mask_ssim_value, grid_path, type_dataset):
        """
        Metodo utilizzato per il salvattaggio delle predizioni durante l'allenamento della rete
        :param epoch: epoca di interesse
        :param id_batch: numero del batch considerato
        :parm batch: contiene il batch considerato
        :param output: predizioni effettuate dalla rete
        :param ssim_value
        :param mask_ssim_value
        :param grid_path: percorso di dove salvare la griglia
        :param type_dataset: tipologia del dataset
        """

        pz_condition = batch[5]  # [batch, 1]
        pz_target = batch[6]  # [batch, 1]
        Ic_image_name = batch[7]  # [batch, 1]
        It_image_name = batch[8]  # [batch, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))

        # GRID: Save griglia di immagini predette
        name_directory = os.path.join(grid_path, type_dataset, str(epoch + 1))
        if not os.path.exists(name_directory):
            os.makedirs(name_directory, exist_ok=False)
        name_grid = os.path.join(name_directory,'Batch_{id_batch}_ssim_{ssim}_mask_ssim_{mask_ssim}.png'.format(
                                     id_batch=id_batch,
                                     ssim=ssim_value,
                                     mask_ssim=mask_ssim_value))
        mean_0 = tf.cast(mean_0, dtype=tf.float16)
        output = self.dataset_module.unprocess_image(output, mean_0, 32765.5)
        utils.save_grid(output, name_grid)

        # File .txt in cui salvo il nome delle immagini di condizione e di target contenute all'interno della griglia
        stack_pairs = np.c_[pz_condition.numpy(), Ic_image_name.numpy(), pz_target.numpy(), It_image_name.numpy()]
        stack_pairs = np.array(
            [[p[0].decode('utf-8'), p[1].decode('utf-8'), p[2].decode('utf-8'), p[3].decode('utf-8')] for p in
             stack_pairs])
        txt_file = 'pz_pair: [<condition>,<target>] \n\n {stack_pair}'.format(stack_pair=np.array2string(stack_pairs))
        file = open(name_directory + '/' + 'Batch_{id_batch}.txt'.format(id_batch=id_batch), "w")
        file.write(txt_file)
        file.close()

    def train_G1(self):
        self.config.load_train_path_G1()

        print("-TRAINING G1")
        print("-Salvo i pesi in: ", self.config.G1_weights_path)
        print("-Salvo le griglie in: ", self.config.G1_grid_path)
        print("-Salvo i logs in: ", self.config.G1_logs_dir_path)

        # - LOGS
        path_history_G1 = os.path.join(self.config.G1_logs_dir_path, 'history_G1.npy')
        history_G1 = {'epoch': 0,
                      'loss_train': np.empty((self.config.G1_epochs)),
                      'ssim_train': np.empty((self.config.G1_epochs)),
                      'mask_ssim_train': np.empty((self.config.G1_epochs)),

                      'loss_valid': np.empty((self.config.G1_epochs)),
                      'ssim_valid': np.empty((self.config.G1_epochs)),
                      'mask_ssim_valid': np.empty((self.config.G1_epochs))}

        # Se esistenti, precarico i logs
        if os.path.exists(path_history_G1):
            print("-Logs preesistenti li precarico")
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
                logs_to_print['mask_ssim_train'][id_batch], I_PT1 = self.__train_on_batch_G1(batch)

                # Grid
                if epoch % self.config.G1_save_grid_ssim_epoch_train == self.config.G1_save_grid_ssim_epoch_train - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT1, logs_to_print['ssim_train'][id_batch],
                                    logs_to_print['mask_ssim_train'][id_batch], self.config.G1_grid_path, type_dataset="train")

                # Logs a schermo
                sys.stdout.write('\rEpoch {epoch} step {id_batch} / {num_batches} -->' \
                                  'loss_G1: {loss_G1:.4f}, ssmi: {ssmi:.4f}, mask_ssmi: {mask_ssmi:.4f}'.format(
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
                logs_to_print['mask_ssim_valid'][id_batch], I_PT1 = self.__valid_on_batch_G1(batch)

                if epoch % self.config.G1_save_grid_ssim_epoch_valid == self.config.G1_save_grid_ssim_epoch_valid - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT1, logs_to_print['ssim_valid'][id_batch],
                                    logs_to_print['mask_ssim_valid'][id_batch], self.config.G1_grid_path, type_dataset="valid")

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
            filepath = os.path.join(self.config.G1_weights_path, name_model)
            self.G1.model.save_weights(filepath)

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

    def __train_on_batch_G1(self, batch):
        """
        Alleno G1 sul batch
        :param batch: contiene il bacth considerato
        """
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

    def __valid_on_batch_G1(self, batch):
        """
        Valido G1 sul batch
        :param batch: contiene il bacth considerato
        """

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
        ssim_value = self.G1.ssim(I_PT1, It, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)
        mask_ssim_value = self.G1.mask_ssim(I_PT1, It, Mt, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)

        return loss_value_G1, ssim_value, mask_ssim_value, I_PT1

    def train_cDCGAN(self):

        # Note: G1 è preaddestrato
        self.config.load_train_path_G1()
        self.config.load_train_path_GAN()
        G1_NAME_WEIGHTS_FILE = os.path.join(self.config.G1_weights_path, self.config.G1_NAME_WEIGHTS_FILE)
        assert os.path.exists(G1_NAME_WEIGHTS_FILE)

        print("-TRAINING cDCGAN")
        print("-Pesi di G1 caricati: ", G1_NAME_WEIGHTS_FILE)
        print("-Salvo i pesi in: ", self.config.GAN_weights_path)
        print("-Salvo le griglie in: ", self.config.GAN_grid_path)
        print("-Salvo i logs in: ", self.config.GAN_logs_dir_path)

        # -History del training
        path_history_GAN = os.path.join(self.config.GAN_logs_dir_path, 'history_GAN.npy')
        history_GAN = {'epoch': 0,
                       'loss_train_G2': np.empty((self.config.GAN_epochs)),
                       'loss_train_D': np.empty((self.config.GAN_epochs)),
                       'loss_train_fake_D': np.empty((self.config.GAN_epochs)),
                       'loss_train_real_D': np.empty((self.config.GAN_epochs)),
                       'ssim_train': np.empty((self.config.GAN_epochs)),
                       'mask_ssim_train': np.empty((self.config.GAN_epochs)),
                       'num_real_I_PT2_train': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'num_real_Ic_train': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'num_real_It_train': np.empty((self.config.GAN_epochs), dtype=np.uint32),

                       'loss_valid_G2': np.empty((self.config.GAN_epochs)),
                       'loss_valid_D': np.empty((self.config.GAN_epochs)),
                       'loss_valid_fake_D': np.empty((self.config.GAN_epochs)),
                       'loss_valid_real_D': np.empty((self.config.GAN_epochs)),
                       'ssim_valid': np.empty((self.config.GAN_epochs)),
                       'mask_ssim_valid': np.empty((self.config.GAN_epochs)),
                       'num_real_I_PT2_valid': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'num_real_Ic_valid': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       'num_real_It_valid': np.empty((self.config.GAN_epochs), dtype=np.uint32),
                       }

        # Se esistenti, precarico i logs
        if os.path.exists(os.path.join(path_history_GAN, 'history_GAN.npy')):
            print("-Logs preesistenti li precarico")
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
        self.G1.model.load_weights(G1_NAME_WEIGHTS_FILE)
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
                             'num_real_I_PT2_train': np.empty((num_batches_train), dtype=np.uint32), #num I_PT2 predette reali dal Discriminatore
                             'num_real_Ic_train': np.empty((num_batches_train), dtype=np.uint32), #num Ic predette reali dal Discriminatore
                             'num_real_It_train': np.empty((num_batches_train), dtype=np.uint32), #num It predette reali dal Discriminatore

                             'loss_values_valid_G2': np.empty((num_batches_valid)),
                             'loss_values_valid_D': np.empty((num_batches_valid)),
                             'loss_values_valid_fake_D': np.empty((num_batches_valid)),
                             'loss_values_valid_real_D': np.empty((num_batches_valid)),
                             'ssim_valid': np.empty((num_batches_valid)),
                             'mask_ssim_valid': np.empty((num_batches_valid)),
                             'num_real_I_PT2_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             'num_real_Ic_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             'num_real_It_valid': np.empty((num_batches_valid), dtype=np.uint32),
                             }

            # TRAIN: iteration
            for id_batch in range(num_batches_train):
                batch = next(train_iterator)
                logs_to_print['loss_values_train_G2'][id_batch], logs_to_print['loss_values_train_D'][id_batch], \
                logs_to_print['loss_values_train_fake_D'][id_batch], logs_to_print['loss_values_train_real_D'][id_batch], \
                logs_to_print['num_real_I_PT2_train'][id_batch], logs_to_print['num_real_Ic_train'][id_batch], \
                logs_to_print['num_real_It_train'][id_batch], logs_to_print['ssim_train'][id_batch], \
                logs_to_print['mask_ssim_train'][id_batch], I_PT2 = \
                    self.__train_on_batch_cDCGAN(id_batch, batch)

                # GRID
                if epoch % self.config.GAN_save_grid_ssim_epoch_train == self.config.GAN_save_grid_ssim_epoch_train - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT2, logs_to_print['ssim_train'][id_batch],
                                    logs_to_print['mask_ssim_train'][id_batch], self.config.GAN_grid_path, type_dataset="train")
                # Logs a schermo
                sys.stdout.write('\rEpoch {epoch} step {id_batch} / {num_batches} --> loss_G2: {loss_G2:2f}, '
                                 'loss_D: {loss_D:2f}, loss_D_fake: {loss_D_fake:2f}, loss_D_real: {loss_D_real:2f}, '
                                 'ssmi: {ssmi:2f}, mask_ssmi: {mask_ssmi:2f}, || '
                                 'numero predette reali:: I_PT2:{I_PT2:1}, Ic:{Ic:1}, It:{It:1} / {total_train}'.format(
                    epoch=epoch + 1,
                    id_batch=id_batch + 1,
                    num_batches=num_batches_train,
                    loss_G2=np.mean(logs_to_print['loss_values_train_G2'][:id_batch + 1]),
                    loss_D=np.mean(logs_to_print['loss_values_train_D'][:id_batch + 1]),
                    loss_D_fake=np.mean(logs_to_print['loss_values_train_fake_D'][:id_batch + 1]),
                    loss_D_real=np.mean(logs_to_print['loss_values_train_real_D'][:id_batch + 1]),
                    ssmi=np.mean(logs_to_print['ssim_train'][:id_batch + 1]),
                    mask_ssmi=np.mean(logs_to_print['mask_ssim_train'][:id_batch + 1]),
                    I_PT2=np.sum(logs_to_print['num_real_I_PT2_train'][:id_batch + 1]),
                    Ic=np.sum(logs_to_print['num_real_Ic_train'][:id_batch + 1]),
                    It=np.sum(logs_to_print['num_real_It_train'][:id_batch + 1]),
                    total_train=self.config.dataset_train_len))
                sys.stdout.flush()

            sys.stdout.write('\nValidazione..\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                batch = next(valid_iterator)
                logs_to_print['loss_values_valid_G2'][id_batch], logs_to_print['loss_values_valid_D'][id_batch], \
                logs_to_print['loss_values_valid_fake_D'][id_batch], logs_to_print['loss_values_valid_real_D'][id_batch], \
                logs_to_print['num_real_I_PT2_valid'][id_batch], logs_to_print['num_real_Ic_valid'][id_batch], \
                logs_to_print['num_real_It_valid'][id_batch], logs_to_print['ssim_valid'][id_batch], \
                logs_to_print['mask_ssim_valid'][id_batch], I_PT2 = self.__valid_on_batch_cDCGAN(batch)

                sys.stdout.write('\r{id_batch} / {total}'.format(id_batch=id_batch + 1, total=num_batches_valid))
                sys.stdout.flush()

                if epoch % self.config.GAN_save_grid_ssim_epoch_valid == self.config.GAN_save_grid_ssim_epoch_valid - 1:
                    self._save_grid(epoch, id_batch, batch, I_PT2, logs_to_print['ssim_valid'][id_batch],
                                    logs_to_print['mask_ssim_valid'][id_batch], self.config.GAN_grid_path, type_dataset="train")

            sys.stdout.write('')
            sys.stdout.write('\r\r'
                'val_loss_G2: {loss_G2:.4f}, val_loss_D: {loss_D:.4f}, val_loss_D_fake: {loss_D_fake:.4f}, '
                'val_loss_D_real: {loss_D_real:.4f}, val_ssmi: {ssmi:.4f}, val_mask_ssmi: {mask_ssmi:.4f} \n\n'
                'numero reali predette: I_PT2:{I_PT2:d}, Ic:{Ic:d}, It:{It:d} / {total_valid}'.format(
                    loss_G2=np.mean(logs_to_print['loss_values_valid_G2']),
                    loss_D=np.mean(logs_to_print['loss_values_valid_D']),
                    loss_D_fake=np.mean(logs_to_print['loss_values_valid_fake_D']),
                    loss_D_real=np.mean(logs_to_print['loss_values_valid_real_D']),
                    ssmi=np.mean(logs_to_print['ssim_valid']),
                    mask_ssmi=np.mean(logs_to_print['mask_ssim_valid']),
                    I_PT2=np.sum(logs_to_print['num_real_I_PT2_valid']),
                    Ic=np.sum(logs_to_print['num_real_Ic_valid']),
                    It=np.sum(logs_to_print['num_real_It_valid']),
                    total_valid=self.config.dataset_valid_len))
            sys.stdout.flush()

            # -CallBacks
            # --Save weights
            # G2
            name_model = "Model_G2_epoch_{epoch:03d}-" \
                         "loss_{loss:.2f}-" \
                         "ssmi_{ssmi:.2f}-" \
                         "mask_ssmi_{mask_ssim:.2f}-" \
                         "I_PT2_{I_PT2:d}-" \
                         "Ic_{Ic:d}-" \
                         "It_{It:d}-" \
                         "val_loss_{val_loss:.2f}-" \
                         "val_ssim_{val_ssim:.2f}-" \
                         "val_mask_ssim_{val_mask_ssim:.2f}-" \
                         "val_I_PT2_{val_I_PT2:d}-" \
                         "val_Ic_{val_Ic:d}-" \
                         "val_It_{val_It:d}.hdf5".format(
                epoch=epoch + 1,
                loss=np.mean(logs_to_print['loss_values_train_G2']),
                ssmi=np.mean(logs_to_print['ssim_train']),
                mask_ssim=np.mean(logs_to_print['mask_ssim_train']),
                I_PT2=int(np.sum(logs_to_print['num_real_I_PT2_train'])),
                Ic=int(np.sum(logs_to_print['num_real_Ic_train'])),
                It=int(np.sum(logs_to_print['num_real_It_train'])),
                val_loss=np.mean(logs_to_print['loss_values_valid_G2']),
                val_ssim=np.mean(logs_to_print['ssim_valid']),
                val_mask_ssim=np.mean(logs_to_print['mask_ssim_valid']),
                val_I_PT2=int(np.sum(logs_to_print['num_real_I_PT2_valid'])),
                val_Ic=int(np.sum(logs_to_print['num_real_Ic_valid'])),
                val_It=int(np.sum(logs_to_print['num_real_It_valid'])),
            )
            filepath = os.path.join(self.config.GAN_weights_path, name_model)
            self.G2.model.save_weights(filepath)

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
            filepath = os.path.join(self.config.GAN_weights_path, name_model)
            self.D.model.save_weights(filepath)

            # --Save logs
            history_GAN['epoch'] = epoch + 1
            history_GAN['loss_train_G2'][epoch] = np.mean(logs_to_print['loss_values_train_G2'])
            history_GAN['loss_train_D'][epoch] = np.mean(logs_to_print['loss_values_train_D'])
            history_GAN['loss_train_fake_D'][epoch] = np.mean(logs_to_print['loss_values_train_fake_D'])
            history_GAN['loss_train_real_D'][epoch] = np.mean(logs_to_print['loss_values_train_real_D'])
            history_GAN['ssim_train'][epoch] = np.mean(logs_to_print['ssim_train'])
            history_GAN['mask_ssim_train'][epoch] = np.mean(logs_to_print['mask_ssim_train'])
            history_GAN['num_real_I_PT2_train'][epoch] = np.sum(logs_to_print['num_real_I_PT2_train'])
            history_GAN['num_real_Ic_train'][epoch] = np.sum(logs_to_print['num_real_Ic_train'])
            history_GAN['num_real_It_train'][epoch] = np.sum(logs_to_print['num_real_It_train'])
            history_GAN['loss_valid_G2'][epoch] = np.mean(logs_to_print['loss_values_valid_G2'])
            history_GAN['loss_valid_D'][epoch] = np.mean(logs_to_print['loss_values_valid_D'])
            history_GAN['loss_valid_fake_D'][epoch] = np.mean(logs_to_print['loss_values_valid_fake_D'])
            history_GAN['loss_valid_real_D'][epoch] = np.mean(logs_to_print['loss_values_valid_real_D'])
            history_GAN['ssim_valid'][epoch] = np.mean(logs_to_print['ssim_valid'])
            history_GAN['mask_ssim_valid'][epoch] = np.mean(logs_to_print['mask_ssim_valid'])
            history_GAN['num_real_I_PT2_valid'][epoch] = np.sum(logs_to_print['num_real_I_PT2_valid'])
            history_GAN['num_real_Ic_valid'][epoch] = np.sum(logs_to_print['num_real_Ic_valid'])
            history_GAN['num_real_It_valid'][epoch] = np.sum(logs_to_print['num_real_It_valid'])
            np.save(os.path.join(path_history_GAN, 'history_GAN.npy'), history_GAN)

            # --Update learning rate
            if epoch % self.config.GAN_lr_update_epoch == self.config.GAN_lr_update_epoch - 1:
                self.G2.optimizer.lr = self.G2.optimizer.lr * self.config.GAN_G2_drop_rate
                self.D.optimizer.lr = self.D.optimizer.lr * self.config.GAN_D_drop_rate
                print("-Aggiornamento Learning rate G2: ", self.G2.optimizer.lr.numpy())
                print("-Aggiornamento Learning rate D: ", self.D.optimizer.lr.numpy())
                print("\n")

            print("#######")

    def __train_on_batch_cDCGAN(self, id_batch, batch):
        """
        Alleno la cDCGAN sul batch
        :param batch: contiene il bacth considerato
        """

        def _tape(loss_function_G2, loss_function_D):
            with tf.GradientTape() as tape:
                I_D = self.G2.prediction(I_PT1, Ic)
                I_PT2 = I_PT1 + I_D  # [batch, 96, 128, 1]

                output_D = self.D.prediction(It, I_PT2, Ic)
                output_D = tf.cast(output_D, dtype=tf.float16)
                D_pos_It, D_neg_I_PT2, D_neg_Ic = tf.split(output_D, 3)  # [batch]

                loss_value_G2 = loss_function_G2(D_neg_I_PT2, I_PT2, It, Mt)
                loss_value_D = loss_function_D(D_pos_It, D_neg_I_PT2, D_neg_Ic)

            return tape, loss_value_G2, loss_value_D, I_PT2, I_D, D_pos_It, D_neg_I_PT2, D_neg_Ic


        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        # G1
        I_PT1 = self.G1.prediction(Ic, Pt)
        # Noise da aggiungere all allenamento del bibranch
        if self.config.ARCHITETURE == "bibranch":
            noise = (np.random.normal(0, 1, I_PT1.shape) * 0.0010) * tf.math.reduce_sum((Pt + 1) / 2, axis=-1).numpy().reshape(I_PT1.shape)
            I_PT1 = tf.add(I_PT1, noise)

        # BACKPROP G2
        I_PT2 = None
        D_neg_I_PT2, D_neg_Ic, D_pos_It = None, None, None
        loss_value_G2, loss_value_D, loss_fake, loss_real = None, None, None, None
        if (id_batch + 1) % 3 == 0:
            g2_tape, loss_value_G2, loss_value_D, I_PT2, I_D, \
            D_pos_It, D_neg_I_PT2, D_neg_Ic = _tape(self.G2.adv_loss, self.D.adv_loss)
            loss_value_D, loss_fake, loss_real = loss_value_D
            self.G2.opt.minimize(loss_value_G2, var_list=self.G2.model.trainable_weights, tape=g2_tape)

        # BACKPROP D
        if not (id_batch + 1) % 3 == 0:
            d_tape, loss_value_G2, loss_value_D, I_PT2, I_D, \
            D_pos_It, D_neg_I_PT2, D_neg_Ic = _tape(self.G2.adv_loss, self.D.adv_loss)
            loss_value_D, loss_fake, loss_real = loss_value_D
            self.D.opt.minimize(loss_value_D, var_list=self.D.model.trainable_weights, tape=d_tape)

        # Metrics
        # - SSIM
        ssim_value = self.G2.ssim(I_PT2, It, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)
        mask_ssim_value = self.G2.mask_ssim(I_PT2, It, Mt, mean_0, mean_1, unprocess_function=self.dataset_module.unprocess_image)

        # - Numero di real predette di I_PT2 dal discriminatore
        np_array_D_neg_I_PT2 = D_neg_I_PT2.numpy()
        num_real_predette_I_PT2_train = np_array_D_neg_I_PT2[np_array_D_neg_I_PT2 > 0]

        # - Numero di real predette di Ic dal discriminatore
        np_array_D_neg_Ic = D_neg_Ic.numpy()
        num_real_predette_Ic_train = np_array_D_neg_Ic[np_array_D_neg_Ic > 0]

        # - Numero di real predette di It dal discriminatore
        np_array_D_pos_It = D_pos_It.numpy()
        num_real_predette_It_train = np_array_D_pos_It[np_array_D_pos_It > 0]

        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               num_real_predette_I_PT2_train.shape[0], num_real_predette_Ic_train.shape[0], \
               num_real_predette_It_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy(), I_PT2

    def __valid_on_batch_cDCGAN(self, batch):
        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

        # G1
        I_PT1 = self.G1.prediction(Ic, Pt)

        # G2
        I_D = self.G2.prediction(I_PT1, Ic)
        I_PT2 = I_PT1 + I_D  # [batch, 96, 128, 1]

        # D
        output_D = self.D.prediction(It, I_PT2, Ic)
        output_D = tf.cast(output_D, dtype=tf.float16)
        D_pos_It, D_neg_I_PT2, D_neg_Ic = tf.split(output_D, 3)  # [batch]

        # Loss
        loss_value_G2 = self.G2.adv_loss(D_neg_I_PT2, I_PT2, It, Ic, Mt)
        loss_value_D, loss_fake, loss_real = self.D.Loss(D_pos_It, D_neg_I_PT2, D_neg_Ic)

        # Metrics
        # - SSIM
        ssim_value = self.G2.m_ssim(I_PT2, It, mean_0, mean_1)
        mask_ssim_value = self.G2.mask_ssim(I_PT2, It, Mt, mean_0, mean_1)

        # - Num real predette di I_PT2 dal discriminatore
        np_array_D_neg_I_PT2 = D_neg_I_PT2.numpy()
        num_real_predette_I_PT2_train = np_array_D_neg_I_PT2[np_array_D_neg_I_PT2 > 0]

        # - Num real predette di Ic dal discriminatore
        np_array_D_neg_Ic = D_neg_Ic.numpy()
        num_real_predette_Ic_train = np_array_D_neg_Ic[np_array_D_neg_Ic > 0]

        # - Num real predette di It dal discriminatore
        np_array_D_pos_It = D_pos_It.numpy()
        num_real_predette_It_train = np_array_D_pos_It[np_array_D_pos_It > 0]

        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), \
               num_real_predette_I_PT2_train.shape[0], num_real_predette_Ic_train.shape[0], \
               num_real_predette_It_train.shape[0], ssim_value.numpy(), mask_ssim_value.numpy(), I_PT2

    def inference_on_test_set_G1(self):
        self.config.load_train_path_G1()
        self.config.load_inference_path_G1()
        G1_NAME_WEIGHTS_FILE = os.path.join(self.config.G1_weights_path, self.config.G1_NAME_WEIGHTS_FILE)
        assert os.path.exists(G1_NAME_WEIGHTS_FILE)

        # Elimino GUI per sovraccarico memoria
        import matplotlib
        matplotlib.use("Agg")

        print("\nINFERENZA DI G1 SU TEST SET")
        print("-Procedo alla predizione su G1")
        print("-Pesi di G1 caricati: ", G1_NAME_WEIGHTS_FILE)
        print("-Le predizioni saranno salvate in: ", self.config.G1_name_dir_test_inference)


        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_test)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)
        dataset_iterator = iter(dataset)

        # Model
        self.G1.model.load_weights(G1_NAME_WEIGHTS_FILE)

        for cnt_img in range(self.config.dataset_test_len):
            sys.stdout.write("\rProcessamento immagine {cnt} / {tot}".format(cnt=cnt_img + 1, tot=self.config.dataset_test_len))
            sys.stdout.flush()
            batch = next(dataset_iterator)
            Ic = batch[0]  # [batch, 96, 128, 1]
            It = batch[1]  # [batch, 96,128, 1]
            Pt = batch[2] # [batch, 96,128, 14]
            Mt = batch[3]  # [batch, 96,128, 1]
            mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
            mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))

            # Predizione
            I_PT1 = self.G1.prediction(Ic, Pt)

            # Unprocess
            Ic = tf.cast(self.dataset_module.unprocess_image(Ic, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
            It = tf.cast(self.dataset_module.unprocess_image(It, mean_1, 32765.5), dtype=tf.uint16)[0].numpy()
            Pt = tf.math.reduce_sum(tf.reshape(tf.math.add(Pt[0], 1, name=None) // 2, [96, 128, 14]),
                                    axis=-1).numpy().reshape(96, 128, 1)
            Pt = tf.cast(Pt, dtype=tf.uint16).numpy()

            Mt = tf.cast(Mt, dtype=tf.uint16)[0].numpy().reshape(96, 128, 1)

            I_PT1 = tf.cast(self.dataset_module.unprocess_image(I_PT1, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()

            # Plot Figure
            fig = plt.figure(figsize=(10, 2))
            columns, rows = 5, 1
            imgs = [I_PT1, Ic, It, Pt, Mt]
            labels = ["I_PT1", "Ic", "It", "Pt", "Mt"]
            for j in range(1, columns * rows + 1):
                sub = fig.add_subplot(rows, columns, j)
                sub.set_title(labels[j - 1])
                plt.imshow(imgs[j - 1], cmap='gray')
            #plt.show()

            #Save figure
            pz_0 = batch[5]  # [batch, 1]
            pz_1 = batch[6]  # [batch, 1]
            name_0 = batch[7]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            pz_0 = pz_0.numpy()[0].decode("utf-8")
            pz_1 = pz_1.numpy()[0].decode("utf-8")
            id_0 = name_0.numpy()[0].decode("utf-8").split('_')[0]  # id dell immagine
            id_1 = name_1.numpy()[0].decode("utf-8").split('_')[0]
            name_img = os.path.join(self.config.G1_name_dir_test_inference, "{id}-{pz_0}_{id_0}-{pz_1}_{id_1}.png".format(
                                        id=cnt_img,
                                        pz_0=pz_0,
                                        pz_1=pz_1,
                                        id_0=id_0,
                                        id_1=id_1))
            plt.savefig(name_img)
            plt.close(fig)
            del fig

    def inference_on_test_set_G2(self):
        self.config.load_train_path_G1()
        self.config.load_train_path_GAN()
        self.config.load_inference_path_GAN()
        G1_NAME_WEIGHTS_FILE = os.path.join(self.config.G1_weights_path, self.config.G1_NAME_WEIGHTS_FILE)
        G2_NAME_WEIGHTS_FILE = os.path.join(self.config.GAN_weights_path, self.config.G2_NAME_WEIGHTS_FILE)
        assert os.path.exists(G1_NAME_WEIGHTS_FILE)
        assert os.path.exists(G2_NAME_WEIGHTS_FILE)

        # Elimino GUI per sovraccarico memoria
        import matplotlib
        matplotlib.use("Agg")

        print("\nINFERENZA DI G2 SU TEST SET")
        print("-Procedo alla predizione su G2")
        print("-Pesi di G1 caricati: ", G1_NAME_WEIGHTS_FILE)
        print("-Pesi di G2 caricati: ", G2_NAME_WEIGHTS_FILE)
        print("-Le predizioni saranno salvate in: ", self.config.GAN_name_dir_test_inference)

        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=self.config.name_tfrecord_test)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)
        dataset_iterator = iter(dataset)

        # Model
        self.G1.model.load_weights(G1_NAME_WEIGHTS_FILE)
        self.G2.model.load_weights(G2_NAME_WEIGHTS_FILE)

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
            Pt = tf.math.reduce_sum(tf.reshape(tf.math.add(Pt[0], 1, name=None), [96, 128, 14]), axis=-1).numpy().reshape(96, 128, 1)# rescale tra [0, 1]
            Pt = tf.cast(Pt, dtype=tf.uint16).numpy()

            Mt = tf.cast(Mt, dtype=tf.uint16)[0].numpy().reshape(96, 128, 1)

            I_PT1 = tf.cast(self.dataset_module.unprocess_image(I_PT1, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
            I_D = tf.cast(self.dataset_module.unprocess_image(I_D, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
            I_PT2 = tf.cast(self.dataset_module.unprocess_image(I_PT2, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()

            # Plot Figure
            fig = plt.figure(figsize=(10, 2))
            columns, rows = 7, 1
            imgs = [I_PT2, I_PT1, I_D, Ic, It, Pt, Mt]
            labels = ["I_PT2", "I_PT1", "I_D", "Ic", "It", "Pt", "Mt"]
            for j in range(1, columns * rows + 1):
                sub = fig.add_subplot(rows, columns, j)
                sub.set_title(labels[j - 1])
                plt.imshow(imgs[j - 1], cmap='gray')
            #plt.show()

            #Save figure
            pz_0 = batch[5]  # [batch, 1]
            pz_1 = batch[6]  # [batch, 1]
            name_0 = batch[7]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            pz_0 = pz_0.numpy()[0].decode("utf-8")
            pz_1 = pz_1.numpy()[0].decode("utf-8")
            id_0 = name_0.numpy()[0].decode("utf-8").split('_')[0]  # id dell immagine
            id_1 = name_1.numpy()[0].decode("utf-8").split('_')[0]
            name_img = os.path.join(self.config.GAN_name_dir_test_inference, "{id}-{pz_0}_{id_0}-{pz_1}_{id_1}.png".format(
                                        id=i,
                                        pz_0=pz_0,
                                        pz_1=pz_1,
                                        id_0=id_0,
                                        id_1=id_1))

            plt.savefig(name_img)
            plt.close(fig)

    # Valutazione metrice IS e FID su uno specifico weight: G1_NAME_WEIGHTS_FILE
    def evaluate_G1(self, name_dataset, dataset_len, analysis_set="test_set", batch_size=10):
        self.config.load_train_path_G1()
        self.config.load_evaluate_path_G1()
        G1_NAME_WEIGHTS_FILE = os.path.join(self.config.G1_weights_path, self.config.G1_NAME_WEIGHTS_FILE)
        assert os.path.exists(G1_NAME_WEIGHTS_FILE)

        print("\nEVALUATE di G1")
        print("-Procedo alla valutazione di G1")
        print("-I file saranno salvati in: ", self.config.G1_evaluation_path)
        print("-Pesi di G1: ", G1_NAME_WEIGHTS_FILE)

        # Dataset
        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=name_dataset)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)

        num_epoch = G1_NAME_WEIGHTS_FILE.split('-')[0].split('_')[-1]
        print("--Valutazione epoca: ", num_epoch)

        # Directory
        path_evaluation = os.path.join(self.config.G1_evaluation_path, analysis_set+'_score_epoch_'+num_epoch) # directory dove salvare i risultati degli score
        path_embeddings = os.path.join(path_evaluation, "inception_embeddings")
        os.makedirs(path_evaluation, exist_ok=True)
        os.makedirs(path_embeddings, exist_ok=True)

        # Model
        self.G1.model.load_weights(G1_NAME_WEIGHTS_FILE)

        # Pipiline score
        utils.evaluation.start([self.G1], iter(dataset), dataset_len, batch_size,
                            dataset_module=self.dataset_module, path_evaluation=path_evaluation,
                            path_embeddings=path_embeddings)

    # Valutazione metrice IS e FID su uno specifico weight: G1_NAME_WEIGHTS_FILE, G2_NAME_WEIGHTS_FILE
    def evaluate_GAN(self, name_dataset, dataset_len, analysis_set="test_set", batch_size=10):
        self.config.load_train_path_G1()
        self.config.load_train_path_GAN()
        self.config.load_evaluate_path_GAN()
        G1_NAME_WEIGHTS_FILE = os.path.join(self.config.G1_weights_path, self.config.G1_NAME_WEIGHTS_FILE)
        G2_NAME_WEIGHTS_FILE = os.path.join(self.config.GAN_weights_path, self.config.G2_NAME_WEIGHTS_FILE)
        assert os.path.exists(G2_NAME_WEIGHTS_FILE)
        assert os.path.exists(G1_NAME_WEIGHTS_FILE)

        print("\nEVALUATE di GAN")
        print("-Procedo alla valutazione di GAN")
        print("-I file saranno salvati in: ", self.config.GAN_evaluation_path)
        print("-I pesi di G1 sono ", G1_NAME_WEIGHTS_FILE)
        print("-I pesi di G2 sono: ", G2_NAME_WEIGHTS_FILE)

        # Dataset
        dataset_unp = self.dataset_module.get_unprocess_dataset(name_tfrecord=name_dataset)
        dataset = self.dataset_module.preprocess_dataset(dataset_unp)
        dataset = dataset.batch(1)

        # Model
        self.G1.model.load_weights(G1_NAME_WEIGHTS_FILE)
        self.G2.model.load_weights(G2_NAME_WEIGHTS_FILE)

        num_epoch_G1 = G1_NAME_WEIGHTS_FILE.split('-')[0].split('_')[-1]
        num_epoch_G2 = G2_NAME_WEIGHTS_FILE.split('-')[0].split('_')[-1]
        print("--Valutazione epoca G1: ", num_epoch_G1)
        print("--Valutazione epoca G2: ", num_epoch_G2)

        # Directory
        path_evaluation = os.path.join(self.config.GAN_evaluation_path, analysis_set + '_score_epochG1_' + num_epoch_G1 +
                                       '_epochG2_' + num_epoch_G2)  # directory dove salvare i risultati degli score
        path_embeddings = os.path.join(path_evaluation, "inception_embeddings")
        os.makedirs(path_evaluation, exist_ok=True)
        os.makedirs(path_embeddings, exist_ok=True)

        # Pipiline score
        utils.evaluation.start([self.G1, self.G2], iter(dataset), dataset_len, batch_size,
                            dataset_module=self.dataset_module,  path_evaluation=path_evaluation,
                            path_embeddings=path_embeddings)

    def plot_history_G1(self):
        self.config.load_train_path_G1()
        path_history_G1 = os.path.join(self.config.G1_logs_dir_path, 'history_G1.npy')
        assert os.path.exists(path_history_G1)
        history_G1 = np.load(path_history_G1, allow_pickle='TRUE')

        epoch = history_G1[()]['epoch']
        loss_train = history_G1[()]['loss_train'][:epoch]
        loss_valid = history_G1[()]['loss_valid'][:epoch]
        ssim_train = history_G1[()]['ssim_train'][:epoch]
        ssim_valid = history_G1[()]['ssim_valid'][:epoch]

        x_axis = np.arange(1, epoch + 1, 1)

        fig, axs = plt.subplots(2)
        axs[0].plot(x_axis, loss_train, label="loss_train")
        axs[0].plot(x_axis, loss_valid, label="loss_valid")
        axs[0].legend()

        axs[1].plot(x_axis, ssim_train, label="ssim_train")
        axs[1].plot(x_axis, ssim_valid, label="ssim_valid")
        axs[1].legend()

        plt.show()

    def plot_history_GAN(self):
        self.config.load_train_path_GAN()
        path_history_GAN = os.path.join(self.config.GAN_logs_dir_path, 'history_GAN.npy')
        assert os.path.exists(path_history_GAN)
        history_GAN = np.load(path_history_GAN, allow_pickle='TRUE')

        epoch = history_GAN[()]['epoch']
        loss_train_G2 = history_GAN[()]['loss_train_G2'][:epoch]
        loss_train_D = history_GAN[()]['loss_train_D'][:epoch]
        loss_train_fake_D = history_GAN[()]['loss_train_fake_D'][:epoch]
        loss_train_real_D = history_GAN[()]['loss_train_real_D'][:epoch]
        loss_valid_G2 = history_GAN[()]['loss_valid_G2'][:epoch]
        loss_valid_D = history_GAN[()]['loss_valid_D'][:epoch]
        loss_valid_fake_D = history_GAN[()]['loss_valid_fake_D'][:epoch]
        loss_valid_real_D = history_GAN[()]['loss_valid_real_D'][:epoch]

        x_axis = np.arange(1, epoch + 1, 1)

        fig, axs = plt.subplots(2)
        axs[0].plot(x_axis, loss_train_G2, label="loss_train_G2")
        axs[0].plot(x_axis, loss_train_D, label="loss_train_D")
        axs[0].plot(x_axis, loss_train_fake_D, label="loss_train_fake_D")
        axs[0].plot(x_axis, loss_train_real_D, label="loss_train_real_D")
        axs[0].legend()

        axs[1].plot(x_axis, loss_valid_G2, label="loss_valid_G2")
        axs[1].plot(x_axis, loss_valid_D, label="loss_valid_D")
        axs[1].plot(x_axis, loss_valid_fake_D, label="loss_valid_fake_D")
        axs[1].plot(x_axis, loss_valid_real_D, label="loss_valid_real_D")
        axs[1].legend()

        plt.show()

    def tsne(self, key_image_interested="test_20"):
        self.config.load_train_path_G1()
        self.config.load_train_path_GAN()
        tsne_path = os.path.join(self.config.OUTPUTS_DIR, "evaluation", "tsne")
        os.makedirs(tsne_path, exist_ok=False)
        G1_NAME_WEIGHTS_FILE = os.path.join(self.config.G1_weights_path, self.config.G1_NAME_WEIGHTS_FILE)
        G2_NAME_WEIGHTS_FILE = os.path.join(self.config.GAN_weights_path, self.config.G2_NAME_WEIGHTS_FILE)
        assert os.path.exists(G2_NAME_WEIGHTS_FILE)
        assert os.path.exists(G1_NAME_WEIGHTS_FILE)

        list_sets = [[self.config.name_tfrecord_train, self.config.dataset_train_len],
                     [self.config.name_tfrecord_valid, self.config.dataset_valid_len],
                     [self.config.name_tfrecord_test, self.config.dataset_test_len]]
        list_perplexity = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 200, 300]

        print("\nEVALUATE di GAN")
        print("-Procedo al calcolo del tsne")
        print("-I file saranno salvati in: ", tsne_path)
        print("-I pesi di G1 sono ", G1_NAME_WEIGHTS_FILE)
        print("-I pesi di G2 sono: ", G2_NAME_WEIGHTS_FILE)
        print("-Le perplexity sono: ", list_perplexity)

        # Model
        self.G1.model.load_weights(G1_NAME_WEIGHTS_FILE)
        self.G2.model.load_weights(G2_NAME_WEIGHTS_FILE)

        # Obtain features
        utils.vgg16_pca_tsne_features.start(list_sets, list_perplexity,
                                            self.G1, self.G2, self.dataset_module,
                                            dir_to_save=tsne_path, save_fig_plot=True, key_image_interested=key_image_interested)