import sys
import os
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
from datasets.Syntetich import Syntetich


def generation():
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    babypose_obj = Syntetich()

    name_dataset = config.name_tfrecord_valid
    name_weights_file = 'Model_G1_epoch_001-loss_0.004598-ssim_0.746547-mask_ssim_0.945029-val_loss_0.001273-val_ssim_0.883418-val_mask_ssim_0.965246.hdf5'
    dataset_len = config.dataset_valid_len
    name_dir_to_save_img = './imgs'

    dataset = babypose_obj.get_unprocess(name_dataset)
    dataset = babypose_obj.get_preprocess(dataset)
    dataset = dataset.batch(1)


    model_G1 = G1.build_model()
    model_G1.load_weights(os.path.join(config.weigths_dir_path, name_weights_file))

    array_pose = []
    array_condizione = []
    array_mean_condizione = []

    iter_ = iter(dataset)
    for i in range(dataset_len):
        batch = next(iter_)
        pose_1 = batch[2]  # [batch, 96,128, 14]
        array_pose.append(pose_1)

    iter_ = iter(dataset)
    for i in range(dataset_len):
        batch = next(iter_)
        if i == 30:
            image_raw_0 = batch[0]  # [batch, 96, 128, 1]
            array_condizione.append(image_raw_0)
            array_mean_condizione.append(tf.reshape(batch[9], (-1, 1, 1, 1)))
            break


    for i in range(len(array_pose)):
        # Predizione
        input_G1 = tf.concat([array_condizione[0], array_pose[i]], axis=-1)
        predizione = model_G1.predict(input_G1)

        image_raw_0 = tf.cast(utils_wgan.unprocess_image(array_condizione[0], array_mean_condizione[0], 32765.5), dtype=tf.uint8)[0].numpy()

        pose_1 = array_pose[i][0]
        pose_1 = tf.math.add(pose_1, 1, name=None) / 2  # rescale tra [0, 1]
        pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
        pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
        pose_1 = tf.cast(pose_1, dtype=tf.uint16).numpy()

        mean_0 = tf.cast(array_mean_condizione[0], dtype=tf.float32)
        predizione = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_0, 32765.5), clip_value_min=0,
                                              clip_value_max=32765), dtype=tf.uint8)[0].numpy()

        # Save Figure
        fig = plt.figure(figsize=(10, 2))
        columns = 3
        rows = 1
        imgs = [predizione, image_raw_0, pose_1]
        labels = ["Predizione", "Immagine di condizione", "Posa desiderata"]
        for j in range(1, columns * rows + 1):
            sub = fig.add_subplot(rows, columns, j)
            sub.set_title(labels[j - 1])
            plt.imshow(imgs[j - 1], cmap='gray')
        name_img = os.path.join(name_dir_to_save_img,
                                "{id}.png".format(
                                    id=i))
        #plt.show()
        plt.savefig(name_img)
        plt.close(fig)


def create_create_video():
    import imageio
    import os
    dir_ = './imgs'
    filenames = os.listdir(dir_)
    with imageio.get_writer('./movie_train.gif', mode='I', fps=5) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(dir_, filename))
            writer.append_data(image)


create_create_video()