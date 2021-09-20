import os
import sys
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from utils import utils_wgan
from utils import grid
from utils.utils_wgan import inception_preprocess_image
from model import G1, G2, Discriminator
from datasets.BabyPose import BabyPose


def calculate_fid_score(embeddings_real, embeddings_fake):
    mu1, sigma1 = embeddings_real.mean(axis=0), np.cov(embeddings_real, rowvar=False)
    mu2, sigma2 = embeddings_fake.mean(axis=0), np.cov(embeddings_fake, rowvar=False)

    diff = mu1 - mu2

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_is_score(model_to_is, embeddings_fake):
    eps = 1e-16
    p_yx = model_to_is.predict(embeddings_fake)  # probabilità condizionale
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)  # probabilità marginale
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))  # kl divergence for each image
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    return np.exp(avg_kl_d)


def save_img(i, name_dir_to_save_img, image_raw_0, image_raw_1, pose_1, mask_1, mean_0, mean_1, predizione, pz_0, pz_1,
             id_0, id_1):
    # Unprocess
    image_raw_0 = utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5).numpy()
    image_raw_0 = tf.cast(image_raw_0, dtype=tf.uint16)[0].numpy()

    image_raw_1 = tf.clip_by_value(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0,
                                   clip_value_max=32765)
    image_raw_1 = tf.cast(image_raw_1, dtype=tf.uint16)[0].numpy()

    pose_1 = pose_1.numpy()[0]
    pose_1 = tf.math.add(pose_1, 1, name=None)  # rescale tra [-1, 1]
    pose_1 = pose_1 / 2
    pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
    pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    pose_1 = tf.cast(pose_1, dtype=tf.float32)

    mask_1 = tf.cast(mask_1, dtype=tf.int16)[0].numpy().reshape(96, 128, 1)

    predizione = tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_0, 32765.5), clip_value_min=0,
                                  clip_value_max=32765)
    predizione = tf.cast(predizione, dtype=tf.uint16)[0].numpy()

    # Save Figure
    fig = plt.figure(figsize=(10, 2))
    columns = 5
    rows = 1
    imgs = [predizione, image_raw_0, pose_1, image_raw_1, mask_1]
    labels = ["Predizione", "Immagine di condizione", "Posa desiderata", "Target", "Maschera posa desiderata"]
    for j in range(1, columns * rows + 1):
        sub = fig.add_subplot(rows, columns, j)
        sub.set_title(labels[j - 1])
        plt.imshow(imgs[j - 1])
    name_img = os.path.join(name_dir_to_save_img,
                            "{id}-{pz_0}_{id_0}-{pz_1}_{id_1}.png".format(
                                id=i,
                                pz_0=pz_0,
                                pz_1=pz_1,
                                id_0=id_0,
                                id_1=id_1))
    plt.savefig(name_img)
    plt.close(fig)


def compute_embeddings_G1(i, inception_model, image_raw_1, predizione, mean_1, mean_0, batch_16_real, batch_16_fake,
                          vettore_embeddings_real, vettore_embeddings_fake):

    batch_16_real[i % 16] = inception_preprocess_image(image_raw_1, mean_1)
    predizione = tf.cast(predizione, dtype=tf.float16)
    batch_16_fake[i % 16] = inception_preprocess_image(predizione, mean_0)

    if (i + 1) % 16 == 0:
        embeddings_real = inception_model.predict(batch_16_real)  # [16,2048]
        embeddings_fake = inception_model.predict(batch_16_fake)  # [16,2048]
        if len(vettore_embeddings_real) > 0:
            vettore_embeddings_real[0] = np.concatenate((vettore_embeddings_real[0], embeddings_real), axis=0)
            vettore_embeddings_fake[0] = np.concatenate((vettore_embeddings_fake[0], embeddings_fake), axis=0)
        else:
            vettore_embeddings_real.append(embeddings_real)
            vettore_embeddings_fake.append(embeddings_fake)

        batch_16_real.fill(1)
        batch_16_fake.fill(1)


def pipeline(model_G1, dataset_aug, dataset_aug_len, name_dir, bool_save_img, bool_loss, bool_ssim, bool_fid, bool_is):
    name_dir_to_save_img = None
    if bool_save_img:
        # Directory
        name_dir_to_save_img = os.path.join(name_dir, "imgs")
        if not os.path.exists(name_dir_to_save_img):
            os.mkdir(name_dir_to_save_img)

    name_dir_to_save_embeddings = None
    fid_score = None
    is_score = None
    inception_model = None
    batch_16_real = None
    batch_16_fake = None
    vettore_embeddings_real = None
    vettore_embeddings_fake = None
    if bool_fid or bool_is:
        # Directory
        name_dir_to_save_embeddings = os.path.join(name_dir, "inception_embeddings")
        if not os.path.exists(name_dir_to_save_embeddings):
            os.mkdir(name_dir_to_save_embeddings)

        # Modello
        inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                            weights="imagenet",
                                                            pooling='avg',
                                                            input_shape=(299, 299, 3))

        # Vettori
        batch_16_real = np.ones((16, 299, 299, 3))
        batch_16_fake = np.ones((16, 299, 299, 3))
        vettore_embeddings_real = []
        vettore_embeddings_fake = []

    ssim_scores = None
    if bool_ssim:
        ssim_scores = np.empty(dataset_aug_len)

    loss_scores = None
    if bool_loss:
        loss_scores = np.empty(dataset_aug_len)

    # Predizione
    for i in range(dataset_aug_len):
        sys.stdout.write('\r')
        sys.stdout.write("Processamento immagine {cnt} / {tot}".format(cnt=i + 1, tot=dataset_aug_len))
        sys.stdout.flush()
        batch = next(dataset_aug)
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

        pz_0 = pz_0.numpy()[0].decode("utf-8")
        pz_1 = pz_1.numpy()[0].decode("utf-8")
        id_0 = name_0.numpy()[0].decode("utf-8").split('_')[0]  # id dell immagine
        id_1 = name_1.numpy()[0].decode("utf-8").split('_')[0]

        # Predizione
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)
        predizione = model_G1.predict(input_G1)

        if bool_save_img:
            save_img(i, name_dir_to_save_img, image_raw_0, image_raw_1, pose_1, mask_1, mean_0, mean_1, predizione,
                     pz_0, pz_1, id_0, id_1)
        if bool_fid or bool_is:
            compute_embeddings_G1(i, inception_model, image_raw_1, predizione, mean_1, mean_0, batch_16_real,
                                  batch_16_fake, vettore_embeddings_real, vettore_embeddings_fake)
        if bool_ssim:
            ssim_scores[i] = G1.m_ssim(predizione, image_raw_1, mean_0, mean_1)
        if bool_loss:
            loss_scores[i] = G1.PoseMaskLoss1(predizione, image_raw_1, image_raw_0, mask_1, mask_0)

    del model_G1
    del batch

    if bool_fid or bool_is:
        np.save(os.path.join(name_dir_to_save_embeddings, "real_2048_embedding.npy"), vettore_embeddings_real[0])
        np.save(os.path.join(name_dir_to_save_embeddings, "fake_2048_embedding.npy"), vettore_embeddings_fake[0])

        if bool_fid:
            fid_score = calculate_fid_score(vettore_embeddings_real[0], vettore_embeddings_fake[0])

        if bool_is:
            inception_model = tf.keras.applications.InceptionV3(include_top=True,
                                                                weights="imagenet",
                                                                pooling='avg',
                                                                input_shape=(299, 299, 3))
            inputs = Input([2048])
            dense_layer = inception_model.layers[-1]
            dense_layer.set_weights(inception_model.layers[-1].get_weights())
            outputs = dense_layer(inputs)
            model_to_is = Model(inputs=inputs, outputs=outputs)
            # model_to_is.summary()

            del inception_model
            is_score = calculate_is_score(model_to_is, vettore_embeddings_fake[0])

    file = open(os.path.join(name_dir, "scores.txt"), "w")
    text = "SSIM: {ssim_value} \nLOSS: {loss_value} \nFID: {fid_value} \nIS: {is_value}".format(
        ssim_value= np.mean(ssim_scores),
        loss_value= np.mean(loss_scores),
        fid_value= fid_score,
        is_value= is_score)
    file.write(text)
    file.close()
    print("\n",text)



if __name__ == "__main__":

    name_dir = 'test_score'  # directory dove salvare i risultati degli score
    name_dataset = "test_aug_dataset.tfrecord"
    name_weights_file = 'Model_G1_epoch_002-loss_0.000353-ssim_0.936256-mask_ssim_0.982756-val_loss_0.000808-val_ssim_0.924695-val_mask_ssim_0.979339.hdf5'

    bool_save_img = True
    bool_ssim = True
    bool_loss = True
    bool_fid = True
    bool_is = True

    # Config file
    Config_file = __import__('1_config_utils')
    config = Config_file.Config()
    config.print_info()
    babypose_obj = BabyPose()

    # Directory
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)

    # Dataset
    dataset_aug_len = 1750
    dataset_aug = babypose_obj.get_unprocess_dataset(name_dataset)
    dataset_aug = babypose_obj.get_preprocess_G1_dataset(dataset_aug)
    # Togliere shugfffle se no non va bene il cnt della save figure
    # dataset_aug = dataset_aug.shuffle(dataset_aug_len // 2, reshuffle_each_iteration=True)
    dataset_aug = dataset_aug.batch(1)
    dataset_aug = iter(dataset_aug)

    # Model
    model_G1 = G1.build_model()
    model_G1.load_weights(os.path.join(config.weigths_path, name_weights_file))

    # Pipiline score
    pipeline(model_G1, dataset_aug, dataset_aug_len, name_dir, bool_save_img, bool_loss, bool_ssim, bool_fid, bool_is)
