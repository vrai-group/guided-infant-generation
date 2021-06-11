import os
import sys
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from utils import utils_wgan

from model import G1, G2, Discriminator
from datasets.market1501 import Market

import matplotlib.pyplot as plt

class PG2(object):

    def __init__(self, config):
        # Trainer.__init__(self, config, data_loader=None)
        self.config = config

        if 'market' in config.dataset.lower():
            self.market_obj = Market()




    def train_G1(self):

        # Preprocess Dataset train
        # TODO CAPIRE SE HA SENSO INVERTIRE E FARE PRIMA LO SHUFFLE E POI LA MAP
        # todo vedere se ha senso specificare che queste operazioni devono avvenire su cpu with tf.device('/cpu:0'):
        dataset_train = self.market_obj.get_unprocess_dataset(config.data_path, "Market1501_train_00000-of-00001.tfrecord")
        dataset_train = dataset_train.shuffle(self.config.dataset_train_len, reshuffle_each_iteration=True)
        dataset_train = self.market_obj.get_preprocess_G1_dataset(dataset_train)
        dataset_train = dataset_train.batch(self.config.batch_size_train)
        dataset_train = dataset_train.repeat(self.config.epochs)  # 100 numeor di epoche
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch
        train_it = iter(dataset_train)

        # Preprocess Dataset valid
        dataset_valid = self.market_obj.get_unprocess_dataset(config.data_path, "Market1501_valid_00000-of-00001.tfrecord")
        dataset_valid = self.market_obj.get_preprocess_G1_dataset(dataset_valid)
        # dataset_valid = dataset_valid.shuffle(self.config.dataset_valid_len, reshuffle_each_iteration=True)
        dataset_valid = dataset_valid.batch(self.config.batch_size_valid)
        dataset_valid = dataset_valid.repeat(self.config.epochs)  # 100 numeor di epoche
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch
        valid_it = iter(dataset_valid)

        # Costruzione modello
        model_g1 = G1.build_model(self.config)
        model_g1.summary()

        # CallBacks
        filepath = os.path.join(self.config.weigths_path, 'Best_Model_{epoch:03d}-{val_loss:2f}-{val_mse:2f}.hdf5')
        checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min', period=1)

        """
        def step_decay(epoch):
            initial_lrate = 0.01
            drop = 0.88
            epochs_drop = 10.0

            lrate = initial_lrate * np.power(drop, np.floor((epoch + 1) / epochs_drop))
            print(lrate)
            return lrate

        learning_rate_decay = LearningRateScheduler(step_decay)
        """

        model_g1.fit(train_it,
                     epochs=self.config.epochs, #100
                     steps_per_epoch=int(self.config.dataset_train_len/self.config.batch_size_train),  #25600/4
                     callbacks=[checkPoint],
                     validation_data = valid_it,
                     validation_steps = int(self.config.dataset_valid_len/self.config.batch_size_valid)
        )

    def train_conditional_GAN(self):

        #Note: G1 è preaddestrato

        # Preprocess Dataset train
        dataset_train = self.market_obj.get_unprocess_dataset(config.data_path, "Market1501_train_00000-of-00001.tfrecord")
        dataset_train = dataset_train.shuffle(self.config.dataset_train_len, reshuffle_each_iteration=True)
        dataset_train = self.market_obj.get_preprocess_GAN_dataset(dataset_train)
        dataset_train = dataset_train.batch(self.config.batch_size_train)
        #dataset_train = dataset_train.repeat(self.config.epochs)  # 100 numeor di epoche
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch

        #Preprocess Dataset valid
        dataset_valid = self.market_obj.get_unprocess_dataset(config.data_path, "Market1501_valid_00000-of-00001.tfrecord")
        dataset_valid = self.market_obj.get_preprocess_GAN_dataset(dataset_valid)
        dataset_valid = dataset_valid.batch(self.config.batch_size_valid)
        #dataset_valid = dataset_valid.repeat(self.config.epochs)  # 100 numeor di epoche
        dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch


        # Carico il modello preaddestrato G1
        self.model_G1 = G1.build_model(self.config)
        self.model_G1.load_weights(os.path.join(self.config.weigths_path, 'Best_Model_001-0.409152-0.158295.hdf5'))

        # Buildo la GAN
        self.model_G2 = G2.build_model(self.config) # architettura Generatore G2
        #self.model_G2.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_0.646448_real_valid_13_real_train_2790.hdf5'))
        self.model_G2.load_weights('./weights/Model_G2_epoch_035-loss_train_2.730875_real_valid_79_real_train_4634.hdf5')
        self.opt_G2 = G2.optimizer() # ottimizzatore
        self.model_D = Discriminator.build_model(self.config)
        #self.model_D.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_2.855738_real_valid_13_real_train_2790.hdf5'))
        self.model_D.load_weights('./weights/Model_D_epoch_035-loss_train_0.659629_real_valid_79_real_train_4634.hdf5')
        self.opt_D = Discriminator.optimizer()

        # Train
        num_batches_train = int(self.config.dataset_train_len / self.config.batch_size_train)  # numero di batches nel dataset di train
        num_batches_valid = int(self.config.dataset_valid_len / self.config.batch_size_valid)  # numero di batches nel dataset di valid

        # Logs da salvare nella cartella logs per ogni epoca
        logs_loss_train_G2 = np.empty((self.config.epochs))
        logs_loss_train_D = np.empty((self.config.epochs))
        logs_loss_train_D_fake = np.empty((self.config.epochs))
        logs_loss_train_D_real = np.empty((self.config.epochs))
        logs_metric_train_G2 = np.empty((self.config.epochs))
        logs_metric_train_D = np.empty((self.config.epochs))

        for epoch in range(self.config.epochs):
            train_it = iter(dataset_train)  # rinizializzo l iteratore sul train dataset
            valid_it = iter(dataset_valid)  # rinizializzo l iteratore sul valid dataset

            mean_loss_G2_train = 0  # calcolo la media della loss ad ogni iterazione sul batch
            mean_loss_D_train = 0  # calcolo la media  della loss ad ogni iterazione sul batch
            mean_loss_D_train_fake = 0
            mean_loss_D_train_real = 0
            cnt_real_predette_train = 0  # counter per le reali (output_G1 + output_G2) predette nel train
            cnt_real_predette_valid = 0 # counter per le reali (output_G1 + output_G2) predette nel valid

            loss_values_train_G2 = np.empty((num_batches_train))  # mi serve per il calcolo della media della loss per ogni batch da printare a schermo
            loss_values_train_D = np.empty((num_batches_train))
            loss_values_train_D_fake = np.empty((num_batches_train))
            loss_values_train_D_real = np.empty((num_batches_train))

            # Train
            for id_batch in range(num_batches_train):
                loss_values_train_G2[id_batch], loss_values_train_D[id_batch], loss_values_train_D_fake[id_batch],loss_values_train_D_real[id_batch], real_predette_train = self._train_step(train_it,id_batch)
                cnt_real_predette_train += real_predette_train
                mean_loss_G2_train = np.mean(loss_values_train_G2[:id_batch])
                mean_loss_D_train = np.mean(loss_values_train_D[:id_batch])
                mean_loss_D_train_fake = np.mean(loss_values_train_D_fake[:id_batch])
                mean_loss_D_train_real = np.mean(loss_values_train_D_real[:id_batch])

                # Logs su schermo
                sys.stdout.write('\r')
                sys.stdout.write('Epoch {epoch} step {id_batch} / {num_batches} --> loss_G2: {loss_G2:2.3f}, loss_D: {loss_D:2.3f}, loss_D_fake: {loss_D_fake:2.3f}, loss_D_real: {loss_D_real:2.3f} real_predette: {real:1} / {total_train}'.format(epoch=epoch + 1, id_batch=id_batch, num_batches=num_batches_train, loss_G2=mean_loss_G2_train, loss_D=mean_loss_D_train, loss_D_fake=mean_loss_D_train_fake, loss_D_real=mean_loss_D_train_real, real=cnt_real_predette_train, total_train=self.config.dataset_train_len))
                sys.stdout.flush()

            sys.stdout.write('\n')
            sys.stdout.write('Validazione..')
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Valid
            for id_batch in range(num_batches_valid):
                 cnt_real_predette_valid += self._valid_step(valid_it)
                 sys.stdout.write('\r')
                 sys.stdout.write('{id_batch} / {total}'.format(id_batch=id_batch, total=num_batches_valid))
                 sys.stdout.flush()

            sys.stdout.write('\r')
            sys.stdout.write('Predette real: {real} / {total}'.format(real=cnt_real_predette_valid, total=self.config.dataset_valid_len))
            sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.write('\n')

            #Save weights
            name_model = "Model_G2_epoch_{epoch:03d}-loss_train_{loss:2f}_real_valid_{real_valid}_real_train_{real_train}.hdf5".format(epoch=epoch+1, loss=mean_loss_G2_train,real_valid=cnt_real_predette_valid,real_train=cnt_real_predette_train)
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_G2.save_weights(filepath)

            name_model = "Model_D_epoch_{epoch:03d}-loss_train_{loss:2f}_real_valid_{real_valid}_real_train_{real_train}.hdf5".format(epoch=epoch+1, loss=mean_loss_D_train,real_valid=cnt_real_predette_valid,real_train=cnt_real_predette_train)
            filepath = os.path.join(self.config.weigths_path, name_model)
            self.model_D.save_weights(filepath)

            #Save logs
            logs_loss_train_G2[epoch] = mean_loss_G2_train
            logs_loss_train_D[epoch] = mean_loss_D_train
            logs_loss_train_D_fake[epoch] = mean_loss_D_train_fake
            logs_loss_train_D_real[epoch] = mean_loss_D_train_real
            np.save('./logs/logs_loss_train_G2epoch_{epoch:03d}.npy'.format(epoch=epoch+1), logs_loss_train_G2[:epoch])
            np.save('./logs/logs_loss_train_D_{epoch:03d}.npy'.format(epoch=epoch+1), logs_loss_train_D[:epoch])
            np.save('./logs/logs_loss_train_D_fake_{epoch:03d}.npy'.format(epoch=epoch+1), logs_loss_train_D_fake[:epoch])
            np.save('./logs/logs_loss_train_D_real_{epoch:03d}.npy'.format(epoch=epoch+1), logs_loss_train_D_real[:epoch])

            #Update learning rate
            if epoch % self.config.lr_update_epoch == self.config.lr_update_epoch - 1:
                self.opt_G2.lr = self.opt_G2.lr * 0.5
                self.opt_D.lr = self.opt_G2.lr * 0.5
                print("Learning rate: ", self.opt_G2.lr)

            # Download from google colab
            if self.config.run_google_colab and (epoch % self.config.download_weight == self.config.download_weight-1):
                os.system('rar a /gdrive/MyDrive/weights_and_logs logs/*')
                #os.system('rar a weights_and_logs weights/Model_G2_epoch_{epoch:03d}-loss_train_*.hdf5'.format(epoch=epoch))
                #os.system('rar a weights_and_logs weights/Model_D_epoch_{epoch:03d}-loss_train_*.hdf5'.format(epoch=epoch))
                os.system('rar a /gdrive/MyDrive/weights_and_logs weights/Model_*_epoch_*-loss_train_*.hdf5')
                print("RAR CREATO\n")

    def _train_step(self, train_it, i):

        batch = next(train_it)
        image_raw_0 = batch[0] #[batch, 128,64, 3]
        image_raw_1 = batch[1] #[batch, 128,64, 3]
        pose_1 = batch[2] #[batch, 128,64, 18]
        mask_1 = batch[3] #[batch, 128,64, 1]

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1) # [batch, 128,64, 21]
        output_G1 = self.model_G1(input_G1) # output_g1 --> [batch, 128, 64, 3]


        with tf.GradientTape() as g2_tape, tf.GradientTape() as d_tape:

            # G2
            input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 128, 64, 6]
            output_G2 = self.model_G2(input_G2) # [batch, 128, 64, 3]
            # s = output_G2.numpy()[0] * 127 + 127
            # cv2.imwrite("D:\\ComputerVision\\PoseGenerationMarket\\test.png", s.astype(np.int32))

            # D
            refined_result = output_G1 + output_G2 # [batch, 128, 64, 3]
            # s = refined_result.numpy()[0] * 127 + 127
            # cv2.imwrite("D:\\ComputerVision\\PoseGenerationMarket\\test.png", s.astype(np.int32))
            input_D = tf.concat([image_raw_1, refined_result, image_raw_0], axis=0) # [batch * 3, 128, 64, 3] --> batch * 3 poichè concateniamo sul primo asse
            output_D = self.model_D(input_D) # [batch * 3, 1]
            output_D = tf.reshape(output_D, [-1]) # [batch*3]
            D_pos_image_raw_1, D_neg_refined_result, D_neg_image_raw_0 = tf.split(output_D,3) # [batch]
            #Come definita in trainer256
            D_z_pos = D_pos_image_raw_1
            D_z_neg = tf.concat([D_neg_refined_result, D_neg_image_raw_0], 0)

            # Loss
            loss_value_G2 = G2.Loss(D_z_neg, refined_result, image_raw_1, mask_1)
            loss_value_D, loss_fake, loss_real = Discriminator.Loss(D_z_pos, D_z_neg, D_neg_image_raw_0)

            # metric
            np_array_D_neg_refined_result = D_neg_refined_result.numpy()
            real_predette_train = np_array_D_neg_refined_result[np_array_D_neg_refined_result > 0]

        # backprop
        self.opt_G2.minimize(loss_value_G2, var_list=self.model_G2.trainable_weights, tape=g2_tape)
        self.opt_D.minimize(loss_value_D, var_list=self.model_D.trainable_weights, tape=d_tape)


        return loss_value_G2.numpy(), loss_value_D.numpy(), loss_fake.numpy(), loss_real.numpy(), real_predette_train.shape[0]

    def _valid_step(self, valid_it):

        batch = next(valid_it)
        image_raw_0 = batch[0] #[batch, 128,64, 3]
        image_raw_1 = batch[1] #[batch, 128,64, 3]
        pose_1 = batch[2] #[batch, 128,64, 18]
        mask_1 = batch[3] #[batch, 128,64, 1]

        # G1
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1) # [batch, 128,64, 21]
        output_G1 = self.model_G1(input_G1) # output_g1 --> [batch, 128, 64, 3]

        # G2
        input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)  # [batch, 128, 64, 6]
        output_G2 = self.model_G2(input_G2)  # [batch, 128, 64, 3]

        # D
        refined_result = output_G1 + output_G2  # [batch, 128, 64, 3]
        output_D = self.model_D(refined_result) # [batch, 1]
        output_D = tf.reshape(output_D, [-1]).numpy() # [batch]

        real_predette = output_D[output_D > 0]


        return real_predette.shape[0] # ritorno il numero di immagini predette come reali

    def predict_G1(self):

        # Preprocess Dataset test
        dataset_test = self.market_obj.get_unprocess_dataset(config.data_path, "Market1501_test_00000-of-00001.tfrecord")
        dataset_test = self.market_obj.get_preprocess_G1_dataset(dataset_test)
        dataset_test = dataset_test.batch(1)
        dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch
        test_it = iter(dataset_test)


        model_g1 = G1(self.config).build_model()
        model_g1.load_weights(os.path.join(self.config.weigths_path, 'Best_Model_001-0.409152-0.158295.hdf5'))


        for cnt in range(int(self.config.dataset_valid_len/1)):

            X, Y = next(test_it)

            if cnt > 500:
                image_raw_0 = X[:,:,:,:3]
                pose_1 = X[:,:,:,3:21]
                image_raw_1 = Y[:, :, :, :3]
                mask_1 = tf.reshape(Y[:, :, :, -1], [-1, 128, 64, 1])
                predizione = model_g1.predict(X, verbose=1)

                #Unprocess
                image_raw_0 = utils_wgan.unprocess_image(image_raw_0, 127.5,127.5)
                image_raw_0 = tf.cast(image_raw_0, dtype=tf.int32)[0]
                image_raw_1 = utils_wgan.unprocess_image(image_raw_1, 127.5, 127.5)
                image_raw_1 = tf.cast(image_raw_1, dtype=tf.int32)[0]
                pose_1 = pose_1.numpy()[0]
                pose_1 = (pose_1.sum(axis=-1) + 1) /2
                pose_1=np.concatenate([image_raw_1[:,:,-1].numpy().reshape((128,64,1)), pose_1.reshape((128,64,1))*255], axis= -1).sum(axis=-1)
                predizione = utils_wgan.unprocess_image(predizione, 127.5, 127.5)
                predizione = tf.cast(predizione, dtype=tf.int32)[0]


                fig = plt.figure(figsize=(10, 10))
                columns = 5
                rows = 1
                imgs = [predizione, image_raw_0, pose_1, image_raw_1, mask_1[0]]
                for i in range(1, columns * rows + 1):
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(imgs[i - 1])
                plt.show()

    def predict_conditional_GAN(self):

        # Preprocess Dataset test
        dataset_test = self.market_obj.get_unprocess_dataset(config.data_path, "Market1501_test_00000-of-00001.tfrecord")
        dataset_test = self.market_obj.get_preprocess_GAN_dataset(dataset_test)
        dataset_test = dataset_test.batch(1)
        dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)  # LASCIO DECIDERE A TENSORFLKOW il numero di memoria corretto per effettuare il prefetch
        test_it = iter(dataset_test)

        # Carico il modello preaddestrato G1
        self.model_G1 = G1.build_model(self.config)
        self.model_G1.load_weights(os.path.join(self.config.weigths_path, 'Best_Model_001-0.409152-0.158295.hdf5'))

        # Carico il modello preaddestrato GAN
        self.model_G2 = G2.build_model(self.config)  # architettura Generatore G2
        # self.model_G2.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_0.646448_real_valid_13_real_train_2790.hdf5'))
        self.model_G2.load_weights('./weights/Model_G2_epoch_035-loss_train_2.730875_real_valid_79_real_train_4634.hdf5')
        self.model_D = Discriminator.build_model(self.config)
        # self.model_D.load_weights(os.path.join(self.config.weigths_path, 'Model_G2_epoch_015-loss_train_2.855738_real_valid_13_real_train_2790.hdf5'))
        self.model_D.load_weights('./weights/Model_D_epoch_035-loss_train_0.659629_real_valid_79_real_train_4634.hdf5')

        for cnt in range(int(self.config.dataset_test_len / 1)):

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
    Config_file = __import__('0_config_utils')
    config = Config_file.Config()

    #prepare_dirs_and_logger(config)

    if config.gpu > -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    pg2 = PG2(config)  # Pose Guided ^2 network

    if config.is_train:
        pg2.train_conditional_GAN()
    else:
        pg2.predict_conditional_GAN()
        




