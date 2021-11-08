import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

from utils.utils_wgan import unprocess_image
from models import G1, G2
from datasets.Syntetich import Syntetich
from keras.applications.vgg16 import VGG16, preprocess_input

def extract_features(feature_extractor, model_G1, model_G2, dataset, dataset_len):

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

        input_G2 = tf.concat([output_G1, image_raw_0], axis=-1)
        output_G2 = model_G2.predict(input_G2)
        predizione = output_G2 + output_G1

        # Unprocess
        image_raw_1_unp = tf.cast(unprocess_image(image_raw_1, mean_1, 32765.5), dtype=tf.uint8)
        image_raw_1_unp = tf.concat([image_raw_1_unp, image_raw_1_unp, image_raw_1_unp], axis=-1)
        image_raw_1_unp = cv2.resize(image_raw_1_unp.numpy()[0], (224, 224))
        image_raw_1_unp = preprocess_input(image_raw_1_unp)
        image_raw_1_unp = np.reshape(image_raw_1_unp, (1, 224, 224, 3))


        predizione_unp = tf.cast(unprocess_image(predizione, mean_0, 32765.5), dtype=tf.uint8)
        predizione_unp = tf.concat([predizione_unp, predizione_unp, predizione_unp], axis=-1)
        predizione_unp = cv2.resize(predizione_unp.numpy()[0], (224, 224))
        predizione_unp = preprocess_input(predizione_unp)
        predizione_unp = np.reshape(predizione_unp, (1,224,224,3))

        features_img1 = feature_extractor.predict(image_raw_1_unp)[0]
        features_og1 = feature_extractor.predict(predizione_unp)[0]

        tot_imgs_features_real.append(features_img1)
        tot_imgs_features_generated.append(features_og1)

    return np.array(tot_imgs_features_real), np.array(tot_imgs_features_generated)

if __name__ == "__main__":
    # Config file
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    babypose_obj = Syntetich()

    name_weights_file_G1 = 'Model_G1_epoch_008-loss_0.000301-ssim_0.929784-mask_ssim_0.979453-val_loss_0.000808-val_ssim_0.911077-val_mask_ssim_0.972699.hdf5'
    name_weights_file_G2 = 'Model_G2_epoch_162-loss_0.69-ssmi_0.93-mask_ssmi_1.00-r_r_5949-im_0_5940-im_1_5948-val_loss_0.70-val_ssim_0.77-val_mask_ssim_0.98.hdf5'
    name_dataset = config.name_tfrecord_train
    dataset_len = config.dataset_train_len

    # Dataset
    dataset = babypose_obj.get_unprocess(name_dataset)
    dataset = babypose_obj.get_preprocess(dataset)
    # Togliere shugfffle se no non va bene il cnt della save figure
    # dataset_aug = dataset_aug.shuffle(dataset_aug_len // 2, reshuffle_each_iteration=True)
    dataset = dataset.batch(1)
    dataset = iter(dataset)

    # Model
    model_G1 = G1.build_model()
    model_G1.load_weights(os.path.join(config.weigths_dir_path, name_weights_file_G1))

    model_G2 = G2.build_model()
    model_G2.load_weights(os.path.join(config.weigths_dir_path, name_weights_file_G2))

    vgg_model = VGG16(include_top=True, weights='imagenet', pooling='max', input_shape=(224, 224, 3), classes=1000)
    # --obtain latent space
    layer = vgg_model.get_layer(name="fc2")
    feature_extractor = Model(inputs=vgg_model.inputs, outputs=layer.output)


    # Pipiline score
    features_real, features_generated = extract_features(feature_extractor, model_G1, model_G2, dataset, dataset_len)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_real = tsne.fit_transform(features_real)
    X_tsne_generated = tsne.fit_transform(features_generated)

    # Create a scatter plot
    plt.scatter(x=X_tsne_real[:, 0], y=X_tsne_real[:, 1], color='red')
    plt.scatter(x=X_tsne_generated[:, 0], y=X_tsne_generated[:, 1], color='blue')

    #plt.show()
    plt.savefig("train_tsne.png")