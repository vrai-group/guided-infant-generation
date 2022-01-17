import os
import sys

import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_for_inception
from skimage.transform import resize


def _calculate_FID_score(embeddings_real, embeddings_fake):
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


def _calculate_IS_score(model_to_is, embeddings_fake):
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


def _inception_preprocess_image(image, mean, unprocess_function):
    def scale_images(images, new_shape):
        new_image = resize(images[0], new_shape, 0)
        new_img = resize(new_image, new_shape, 0)
        v = np.empty((1, new_shape[0], new_shape[1], new_shape[2]))
        v[0] = new_img
        return v

    image = tf.reshape(image, [-1, 96, 128, 1])
    image = tf.cast(tf.cast(unprocess_function(image, mean, 32765.5), dtype=tf.uint8), dtype=tf.float32)

    image_3channel = tf.concat([image, image, image], axis=-1)
    image_3channel = scale_images(image_3channel, (299, 299, 3))
    image_3channel_p = preprocess_input_for_inception(image_3channel)

    return image_3channel_p


def _save_img(cnt_img, name_dir_to_save_img, Ic, It, Pt, Mt, mean_0, mean_1, I_PT1, pz_0, pz_1, id_0, id_1,
              dataset_module):
    # Unprocess
    Ic = tf.cast(dataset_module.unprocess_image(Ic, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()
    It = tf.cast(dataset_module.unprocess_image(It, mean_1, 32765.5), dtype=tf.uint16)[0].numpy()

    Mt = tf.cast(Mt, dtype=tf.uint16)[0].numpy().reshape(96, 128, 1)

    I_PT1 = tf.cast(dataset_module.unprocess_image(I_PT1, mean_0, 32765.5), dtype=tf.uint16)[0].numpy()

    Pt = Pt[0]
    Pt = tf.math.add(Pt, 1, name=None) / 2  # rescale tra [0, 1]
    Pt = tf.reshape(Pt, [96, 128, 14]) * 255
    Pt = tf.math.reduce_sum(Pt, axis=-1).numpy().reshape(96, 128, 1)
    Pt = tf.cast(Pt, dtype=tf.uint16).numpy()

    # Save Figure
    fig = plt.figure(figsize=(10, 2))
    columns, rows = 5, 1
    imgs = [I_PT1, Ic, Pt, It, Mt]
    labels = ["I_PT1", "Ic", "Pt", "It", "Mt"]
    for j in range(1, columns * rows + 1):
        sub = fig.add_subplot(rows, columns, j)
        sub.set_title(labels[j - 1])
        plt.imshow(imgs[j - 1], cmap='gray')
    name_img = os.path.join(name_dir_to_save_img, "{id}-{pz_0}_{id_0}-{pz_1}_{id_1}.png".format(
        id=cnt_img,
        pz_0=pz_0,
        pz_1=pz_1,
        id_0=id_0,
        id_1=id_1))
    plt.show()
    plt.savefig(name_img)
    plt.close(fig)


def _compute_embeddings_inception(cnt_embeddings, inception_model, batch_size,
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


def start(model_G1, dataset, len_dataset, batch_size, dataset_module, bool_save_img, name_dir_to_save_evaluation):
    # Directory dove salvare le immagine di predizione
    name_dir_to_save_img = None
    if bool_save_img:
        name_dir_to_save_img = os.path.join(name_dir_to_save_evaluation, "imgs")
        if not os.path.exists(name_dir_to_save_img):
            os.mkdir(name_dir_to_save_img)

    # Directory dove salvare gli embeddings
    name_dir_to_save_embeddings = os.path.join(name_dir_to_save_evaluation, "inception_embeddings")
    if not os.path.exists(name_dir_to_save_embeddings):
        os.mkdir(name_dir_to_save_embeddings)

    # Modello Inception
    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling='avg',
                                                        input_shape=(299, 299, 3))

    # Vettori
    input_inception_real = np.empty((batch_size, 299, 299, 3))  # conterrà le immagini It da dare in input all inception
    input_inception_fake = np.empty((batch_size, 299, 299, 3))  # conterrà le immagini I_PT1 da dare in input all inception
    vettore_embeddings_real = np.empty((len_dataset, 2048))  # conterrà gli embeddings dell'inception per i reali
    vettore_embeddings_fake = np.empty((len_dataset, 2048)) #  conterrà gli embeddings dell'inception per le generate

    # Vettori Mask
    input_inception_mask_real = np.empty((batch_size, 299, 299, 3)) # conterrà le immagini It * Mt da dare in input all inception
    input_inception_mask_fake = np.empty((batch_size, 299, 299, 3)) # conterrà le immagini I_PT1 * Mt da dare in input all inception
    vettore_embeddings_mask_real = np.empty((len_dataset, 2048))
    vettore_embeddings_mask_fake = np.empty((len_dataset, 2048))
    cnt_embeddings = 0

    ########## SSIM SCORE
    ssim_scores = np.empty(len_dataset)
    mask_ssim_scores = np.empty(len_dataset)

    ########## LOSS SCORE
    loss_scores = np.empty(len_dataset)

    # Predizione
    for cnt_img in range(len_dataset):
        sys.stdout.write("\rProcessamento immagine {cnt} / {tot}".format(cnt=cnt_img + 1, tot=len_dataset))
        sys.stdout.flush()
        batch = next(dataset)
        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]

        pz_0 = batch[5].numpy()[0].decode("utf-8")
        pz_1 = batch[6].numpy()[0].decode("utf-8")
        id_0 = batch[7].numpy()[0].decode("utf-8").split('_')[0]
        id_1 = batch[8].numpy()[0].decode("utf-8").split('_')[0]
        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))
        mask_image_raw_1 = It * Mt

        # Predizione G1
        I_PT1 = model_G1.preiction(Ic, Pt)
        mask_predizione = I_PT1 * Mt

        # Salvataggio immagine
        if bool_save_img:
            _save_img(cnt_img, name_dir_to_save_img, Ic, It, Pt, Mt, mean_0, mean_1, I_PT1, pz_0, pz_1, id_0, id_1,
                      dataset_module)

        # Preprocesso immagini per Inception model
        input_inception_real[cnt_img % batch_size] = _inception_preprocess_image(It, mean_1,
                                                                                 unprocess_function=dataset_module.unprocess_image)
        input_inception_fake[cnt_img % batch_size] = _inception_preprocess_image(tf.cast(I_PT1, dtype=tf.float16),
                                                                                 mean_0,
                                                                                 unprocess_function=dataset_module.unprocess_image)

        input_inception_mask_real[cnt_img % batch_size] = _inception_preprocess_image(mask_image_raw_1, mean_1,
                                                                                      unprocess_function=dataset_module.unprocess_image)
        input_inception_mask_fake[cnt_img % batch_size] = _inception_preprocess_image(
            tf.cast(mask_predizione, dtype=tf.float16), mean_0, unprocess_function=dataset_module.unprocess_image)

        if (cnt_img + 1) % batch_size == 0:
            _compute_embeddings_inception(cnt_embeddings, inception_model, batch_size,
                                          input_inception_real, input_inception_mask_real,
                                          input_inception_fake, input_inception_mask_fake,
                                          vettore_embeddings_real, vettore_embeddings_mask_real,
                                          vettore_embeddings_fake, vettore_embeddings_mask_fake)
            cnt_embeddings += 1
            input_inception_real.fill(0)
            input_inception_fake.fill(0)
            input_inception_mask_real.fill(0)
            input_inception_mask_fake.fill(0)
    """
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
    
    """