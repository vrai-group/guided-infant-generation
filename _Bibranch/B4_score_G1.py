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
from models import G1_Bibranch as G1
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
    p_yx = model_to_is.predict(embeddings_fake)  # probabilità condizionale [batch, 1000]
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)  # probabilità marginale [1, 1000]
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))  # kl divergence for each image [batch, 1000]
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)  # [bacth,1]
    # average over images
    avg_kl_d = np.mean(sum_kl_d)  # [1,1]
    # undo the logs
    return np.exp(avg_kl_d)


def save_img(i, name_dir_to_save_img, image_raw_0, image_raw_1, pose_1, mask_1, mean_0, mean_1, predizione, pz_0, pz_1,
             id_0, id_1):
    # Unprocess
    image_raw_0 = tf.cast(utils_wgan.unprocess_image(image_raw_0, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
    image_raw_1 = tf.cast(utils_wgan.unprocess_image(image_raw_1, mean_1, 32765.5), dtype=tf.uint16)[0].numpy()

    pose_1 = pose_1[0]
    pose_1 = tf.math.add(pose_1, 1, name=None) / 2 # rescale tra [0, 1]
    pose_1 = tf.reshape(pose_1, [96, 128, 14]) * 255
    pose_1 = tf.math.reduce_sum(pose_1, axis=-1).numpy().reshape(96, 128, 1)
    pose_1 = tf.cast(pose_1, dtype=tf.uint16).numpy()

    mask_1 = tf.cast(mask_1, dtype=tf.uint16)[0].numpy().reshape(96, 128, 1)

    predizione = tf.cast(tf.clip_by_value(utils_wgan.unprocess_image(predizione, mean_0, 32765.5), clip_value_min=0,
                                  clip_value_max=32765), dtype=tf.uint16)[0].numpy()

    # Save Figure
    fig = plt.figure(figsize=(10, 2))
    columns = 5
    rows = 1
    imgs = [predizione, image_raw_0, pose_1, image_raw_1, mask_1]
    labels = ["Predizione", "Immagine di condizione", "Posa desiderata", "Target", "Maschera posa desiderata"]
    for j in range(1, columns * rows + 1):
        sub = fig.add_subplot(rows, columns, j)
        sub.set_title(labels[j - 1])
        plt.imshow(imgs[j - 1],cmap='gray')
    name_img = os.path.join(name_dir_to_save_img,
                            "{id}-{pz_0}_{id_0}-{pz_1}_{id_1}.png".format(
                                id=i,
                                pz_0=pz_0,
                                pz_1=pz_1,
                                id_0=id_0,
                                id_1=id_1))
    plt.show()
    plt.savefig(name_img)
    plt.close(fig)


def compute_embeddings_G1(cnt_embeddings, inception_model, batch_size,
                          input_inception_real, input_inception_mask_real,
                          input_inception_fake, input_inception_mask_fake,
                          vettore_embeddings_real, vettore_embeddings_mask_real,
                          vettore_embeddings_fake, vettore_embeddings_mask_fake):
    start = cnt_embeddings * batch_size
    end = start + batch_size
    vettore_embeddings_real[start:end] = inception_model.predict(input_inception_real)  # [batch,2048]
    vettore_embeddings_fake[start:end] = inception_model.predict(input_inception_fake)
    vettore_embeddings_mask_real[start:end] = inception_model.predict(input_inception_mask_real)
    vettore_embeddings_mask_fake[start:end] = inception_model.predict(input_inception_mask_fake)


def pipeline(model_G1, dataset_aug, dataset_aug_len, name_dir, batch_size, bool_save_img):
    name_dir_to_save_img = None
    if bool_save_img:
        # Directory
        name_dir_to_save_img = os.path.join(name_dir, "imgs")
        if not os.path.exists(name_dir_to_save_img):
            os.mkdir(name_dir_to_save_img)

    ########## FID-IS SCORE
    name_dir_to_save_embeddings = os.path.join(name_dir, "inception_embeddings")
    if not os.path.exists(name_dir_to_save_embeddings):
        os.mkdir(name_dir_to_save_embeddings)

    # Modello
    inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights="imagenet",
                                                        pooling='avg',
                                                        input_shape=(299, 299, 3))

    # Vettori
    input_inception_real = np.empty((batch_size, 299, 299, 3))
    input_inception_fake = np.empty((batch_size, 299, 299, 3))
    vettore_embeddings_real = np.empty((dataset_aug_len, 2048))
    vettore_embeddings_fake = np.empty((dataset_aug_len, 2048))

    # Vettori Mask
    input_inception_mask_real = np.empty((batch_size, 299, 299, 3))
    input_inception_mask_fake = np.empty((batch_size, 299, 299, 3))
    vettore_embeddings_mask_real = np.empty((dataset_aug_len, 2048))
    vettore_embeddings_mask_fake = np.empty((dataset_aug_len, 2048))
    cnt_embeddings = 0

    ########## SSIM SCORE
    ssim_scores = np.empty(dataset_aug_len)
    mask_ssim_scores = np.empty(dataset_aug_len)

    ########## LOSS SCORE
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
        mask_image_raw_1 = image_raw_1 * mask_1
        mask_predizione = None

        pz_0 = pz_0.numpy()[0].decode("utf-8")
        pz_1 = pz_1.numpy()[0].decode("utf-8")
        id_0 = name_0.numpy()[0].decode("utf-8").split('_')[0]  # id dell immagine
        id_1 = name_1.numpy()[0].decode("utf-8").split('_')[0]

        # Predizione
        input_G1 = tf.concat([image_raw_0, pose_1], axis=-1)
        predizione = model_G1.predict(input_G1)
        mask_predizione = predizione * mask_1

        if bool_save_img:
            save_img(i, name_dir_to_save_img, image_raw_0, image_raw_1, pose_1, mask_1, mean_0, mean_1, predizione,
                     pz_0, pz_1, id_0, id_1)

        ### Ottengo embeddings
        input_inception_real[i % batch_size] = inception_preprocess_image(image_raw_1, mean_1)
        input_inception_fake[i % batch_size] = inception_preprocess_image(tf.cast(predizione, dtype=tf.float16), mean_0)

        input_inception_mask_real[i % batch_size] = inception_preprocess_image(mask_image_raw_1, mean_1)
        input_inception_mask_fake[i % batch_size] = inception_preprocess_image(
            tf.cast(mask_predizione, dtype=tf.float16), mean_0)

        if (i + 1) % batch_size == 0:
            compute_embeddings_G1(cnt_embeddings, inception_model, batch_size,
                                  input_inception_real, input_inception_mask_real,
                                  input_inception_fake, input_inception_mask_fake,
                                  vettore_embeddings_real, vettore_embeddings_mask_real,
                                  vettore_embeddings_fake, vettore_embeddings_mask_fake)
            cnt_embeddings += 1
            input_inception_real.fill(0)
            input_inception_fake.fill(0)
            input_inception_mask_real.fill(0)
            input_inception_mask_fake.fill(0)

        ssim_scores[i] = G1.m_ssim(predizione, image_raw_1, mean_0, mean_1)
        mask_ssim_scores[i] = G1.mask_ssim(predizione, image_raw_1, mask_1, mean_0, mean_1)

        loss_scores[i] = G1.PoseMaskLoss1(predizione, image_raw_1, image_raw_0, mask_1, mask_0)

    del model_G1
    del batch

    np.save(os.path.join(name_dir_to_save_embeddings, "real_2048_embedding.npy"), vettore_embeddings_real)
    np.save(os.path.join(name_dir_to_save_embeddings, "fake_2048_embedding.npy"), vettore_embeddings_fake)
    np.save(os.path.join(name_dir_to_save_embeddings, "mask_real_2048_embedding.npy"), vettore_embeddings_mask_real)
    np.save(os.path.join(name_dir_to_save_embeddings, "mask_fake_2048_embedding.npy"), vettore_embeddings_mask_fake)

    fid_score = calculate_fid_score(vettore_embeddings_real, vettore_embeddings_fake)
    mask_fid_score = calculate_fid_score(vettore_embeddings_mask_real, vettore_embeddings_mask_fake)

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
    is_score = calculate_is_score(model_to_is, vettore_embeddings_fake)
    is_score_real = calculate_is_score(model_to_is, vettore_embeddings_real)

    mask_is_score = calculate_is_score(model_to_is, vettore_embeddings_mask_fake)
    mask_is_score_real = calculate_is_score(model_to_is, vettore_embeddings_mask_real)

    file = open(os.path.join(name_dir, "scores.txt"), "w")
    text = "\nLOSS: {loss_value} " \
           "\nSSIM: {ssim_value} " \
           "\nFID: {fid_value} " \
           "\nIS: {is_value} " \
           "\nIS_real: {is_value_real}" \
           "\n\n" \
           "\nMASK_SSIM:{mask_ssim_value} " \
           "\nMASK_FID: {mask_fid_value} " \
           "\nMASK_IS: {mask_is_value} " \
           "\nMASK_IS_real: {mask_is_value_real}".format(
        ssim_value=np.mean(ssim_scores),
        mask_ssim_value=np.mean(mask_ssim_scores),
        loss_value=np.mean(loss_scores),
        fid_value=fid_score,
        mask_fid_value=mask_fid_score,
        is_value=is_score,
        mask_is_value=mask_is_score,
        is_value_real=is_score_real,
        mask_is_value_real=mask_is_score_real)
    file.write(text)
    file.close()
    print("\n", text)


if __name__ == "__main__":
    # Config file
    Config_file = __import__('B1_config_utils')
    config = Config_file.Config()
    babypose_obj = BabyPose()

    for w in os.listdir('./weights'):
        num = w.split('-')[0].split('_')[4]
        name_dir = 'test_score_epoca'+num  # directory dove salvare i risultati degli score
        name_dataset = config.name_tfrecord_test
        #name_weights_file = 'Model_G1_epoch_002-loss_0.000704-ssim_0.913195-mask_ssim_0.975810-val_loss_0.000793-val_ssim_0.912054-val_mask_ssim_0.974530.hdf5'
        name_weights_file = w
        bool_save_img = True
        batch_size = 10
        dataset_len = config.dataset_test_len

        # Directory
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)

        # Dataset
        dataset = babypose_obj.get_unprocess_dataset(name_dataset)
        dataset = babypose_obj.get_preprocess_G1_dataset(dataset)
        # Togliere shugfffle se no non va bene il cnt della save figure
        # dataset_aug = dataset_aug.shuffle(dataset_aug_len // 2, reshuffle_each_iteration=True)
        dataset = dataset.batch(1)
        dataset = iter(dataset)

        # Model
        model_G1 = G1.build_model()
        model_G1.load_weights(os.path.join(config.weigths_dir_path, name_weights_file))

        # Pipiline score
        pipeline(model_G1, dataset, dataset_len, name_dir, batch_size, bool_save_img=bool_save_img)
