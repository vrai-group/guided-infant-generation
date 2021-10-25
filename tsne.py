from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import preprocess_input



from utils.utils_wgan import unprocess_image
from model import G1, G2, Discriminator
from datasets.BabyPose import BabyPose
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

def extract_features(feature_extractor, model_G1, dataset, dataset_len):

    tot_imgs_features_real = []  # qui salvo le features su tutto il dataset
    tot_imgs_features_generated = []
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

        # Unprocess
        image_raw_1_unp = tf.cast(unprocess_image(image_raw_1, mean_1, 32765.5), dtype=tf.uint8)
        image_raw_1_unp = tf.concat([image_raw_1_unp, image_raw_1_unp, image_raw_1_unp], axis=-1)
        image_raw_1_unp = cv2.resize(image_raw_1_unp.numpy()[0], (224, 224))
        image_raw_1_unp = preprocess_input(image_raw_1_unp)
        image_raw_1_unp = np.reshape(image_raw_1_unp, (1, 224, 224, 3))


        output_G1_unp = tf.cast(unprocess_image(output_G1, mean_0, 32765.5), dtype=tf.uint8)
        output_G1_unp = tf.concat([output_G1_unp, output_G1_unp, output_G1_unp], axis=-1)
        output_G1_unp = cv2.resize(output_G1_unp.numpy()[0], (224, 224))
        output_G1_unp = preprocess_input(output_G1_unp)
        output_G1_unp = np.reshape(output_G1_unp, (1,224,224,3))

        features_img1 = feature_extractor.predict(image_raw_1_unp)[0]
        features_og1 = feature_extractor.predict(output_G1_unp)[0]

        tot_imgs_features_real.append(features_img1)
        tot_imgs_features_generated.append(features_og1)

    return np.array(tot_imgs_features_real), np.array(tot_imgs_features_generated)

if __name__ == "__main__":
    # Config file
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    babypose_obj = BabyPose()

    name_weights_file_G1 = 'Model_G1_epoch_008-loss_0.000301-ssim_0.929784-mask_ssim_0.979453-val_loss_0.000808-val_ssim_0.911077-val_mask_ssim_0.972699.hdf5'
    name_dataset = config.name_tfrecord_train
    dataset_len = config.dataset_train_len

    # Dataset
    dataset = babypose_obj.get_unprocess_dataset(name_dataset)
    dataset = babypose_obj.get_preprocess_G1_dataset(dataset)
    # Togliere shugfffle se no non va bene il cnt della save figure
    # dataset_aug = dataset_aug.shuffle(dataset_aug_len // 2, reshuffle_each_iteration=True)
    dataset = dataset.batch(1)
    dataset = iter(dataset)

    # Model
    model_G1 = G1.build_model()
    model_G1.load_weights(os.path.join(config.weigths_path, name_weights_file_G1))

    vgg_model = VGG16(include_top=True, weights='imagenet', pooling='max', input_shape=(224, 224, 3), classes=1000)
    # --obtain latent space
    layer = vgg_model.get_layer(name="fc2")
    feature_extractor = Model(inputs=vgg_model.inputs, outputs=layer.output)


    # Pipiline score
    features_real, features_generated = extract_features(feature_extractor, model_G1, dataset, dataset_len)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_real = tsne.fit_transform(features_real)
    X_tsne_generated = tsne.fit_transform(features_generated)

    # Create a scatter plot
    plt.scatter(x=X_tsne_real[:, 0], y=X_tsne_real[:, 1], color='red')
    plt.scatter(x=X_tsne_generated[:, 0], y=X_tsne_generated[:, 1], color='blue')

    # Change chart background color
    #fig.update_layout(dict(plot_bgcolor='white'))

    # Update axes lines
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
    #                  zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
    #                  showline=True, linewidth=1, linecolor='black')
    #
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
    #                  zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
    #                  showline=True, linewidth=1, linecolor='black')
    #
    # # Set figure title
    # fig.update_layout(title_text="t-SNE")
    #
    # # Update marker size
    # fig.update_traces(marker=dict(size=3))

    #plt.show()
    plt.savefig("train_tsne.png")