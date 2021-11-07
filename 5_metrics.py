from sklearn.metrics import mean_squared_log_error as rmsle
import os
import sys

import cv2
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from utils import utils_wgan
from utils import grid
from utils.utils_wgan import inception_preprocess_image
from models import G1, G2, Discriminator
from _Bibranch.datasets.BabyPose import BabyPose

def pipeline(model_G1, model_G2,dataset_len, dataset):
    # Predizione

    tot = []

    for i in range(dataset_len):
        sys.stdout.write('\r')
        sys.stdout.write("Processamento immagine {cnt} / {tot}".format(cnt=i + 1, tot=dataset_len))
        sys.stdout.flush()
        batch = next(dataset)
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
        mask_image_raw_1 = image_raw_1 * mask_1
        mask_predizione = None

        pz_0 = pz_0.numpy()[0].decode("utf-8")
        pz_1 = pz_1.numpy()[0].decode("utf-8")
        id_0 = name_0.numpy()[0].decode("utf-8").split('_')[0]  # id dell immagine
        id_1 = name_1.numpy()[0].decode("utf-8").split('_')[0]

        # Predizione
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)
        output_G1 = model_G1.predict(input_G1)
        predizione = output_G1

        # input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)
        # output_G2 = model_G2.predict(input_G2)
        # predizione = output_G2 + output_G1

        #RMSLE
        image_raw_1 = utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5)
        image_raw_1 = image_raw_1[0].numpy().reshape(96,128)
        predizione = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        predizione = predizione[0].numpy().reshape(96,128)
        value = rmsle(image_raw_1, predizione, sample_weight=None, multioutput='uniform_average', squared=True)
        tot.append(value)
    mean = np.mean(np.asarray(tot))
    print()
    print(mean)




if __name__ == "__main__":
    # Config file
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    babypose_obj = BabyPose()

    name_weights_file_G1 = 'Model_G1_epoch_008-loss_0.000301-ssim_0.929784-mask_ssim_0.979453-val_loss_0.000808-val_ssim_0.911077-val_mask_ssim_0.972699.hdf5'
    for w in os.listdir('./weights/G2'):
        name_weights_file_G2 = w
        num = name_weights_file_G2.split('-')[0].split('_')[3]
        name_dir = 'test_score_epoca' + num  # directory dove salvare i risultati degli score
        name_dataset = config.name_tfrecord_test
        dataset_len = config.dataset_test_len

        # Dataset
        dataset = babypose_obj.get_unprocess_dataset(name_dataset)
        dataset = babypose_obj.get_preprocess_G1_Bibranch_dataset(dataset)
        # Togliere shugfffle se no non va bene il cnt della save figure
        # dataset_aug = dataset_aug.shuffle(dataset_aug_len // 2, reshuffle_each_iteration=True)
        dataset = dataset.batch(1)
        dataset = iter(dataset)

        # Model
        model_G1 = G1.build_model()
        model_G1.load_weights(os.path.join(config.weigths_path, name_weights_file_G1))

        model_G2 = G2.build_model()
        model_G2.summary()
        model_G2.load_weights(os.path.join(config.weigths_path, "G2", name_weights_file_G2))

        # Pipiline score
        pipeline(model_G1, model_G2, dataset_len, dataset)