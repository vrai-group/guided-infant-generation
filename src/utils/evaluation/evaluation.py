import os
import sys

import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_for_inception
from skimage.transform import resize

# INCEPTION_IMG_W, INCEPTION_IMG_H, INCEPTION_IMG_CH = 299, 299, 3
# 2048 = 2048

def _inception_preprocess_image(image, mean, unprocess_function):
    def scale_images(images, new_shape):
        new_image = resize(images[0], new_shape, 0)
        new_img = resize(new_image, new_shape, 0)
        v = np.empty((1, new_shape[0], new_shape[1], new_shape[2]))
        v[0] = new_img
        return v

    image = tf.reshape(image, [-1, 96, 128, 1])
    image = tf.cast(tf.cast(unprocess_function(image, mean), dtype=tf.uint8), dtype=tf.float32)

    image_3channel = tf.concat([image, image, image], axis=-1)
    image_3channel = scale_images(image_3channel, (299, 299, 3))
    image_3channel_p = preprocess_input_for_inception(image_3channel)

    return image_3channel_p

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
    p_yx = model_to_is.predict(embeddings_fake)  # probabilit√† condizionale [batch, 1000]
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)  # probabilit√† marginale [1, 1000]
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))  # kl divergence for each image [batch, 1000]
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)  # [bacth,1]
    # average over images
    avg_kl_d = np.mean(sum_kl_d)  # [1,1]
    # undo the logs
    return np.exp(avg_kl_d)


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

def start(models, dataset, len_dataset, batch_size, dataset_module, path_evaluation, path_embeddings):
    G1, G2 = None, None
    if len(models) == 1:
        G1 = models[0]
    elif len(models) == 2:
        G1 = models[0]
        G2 = models[1]

    # Modello Inception
    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling='avg',
                                                        input_shape=(299, 299, 3))

    # Vettori immagini
    input_inception_real = np.empty((batch_size, 299, 299, 3))  # conterr√† le immagini It da dare in input all inception
    input_inception_fake = np.empty((batch_size, 299, 299, 3))  # conterr√† le immagini I_PT1 da dare in input all inception
    input_inception_mask_real = np.empty((batch_size, 299, 299, 3))  # conterr√† le immagini It * Mt da dare in input all inception
    input_inception_mask_fake = np.empty((batch_size, 299, 299, 3))  # conterr√† le immagini I_PT1 * Mt da dare in input all inception

    # Vettori embeddings
    vettore_embeddings_real = np.empty((len_dataset, 2048))  # conterr√† gli embeddings dell'inception per i reali
    vettore_embeddings_fake = np.empty((len_dataset, 2048))  # conterr√† gli embeddings dell'inception per le generate
    vettore_embeddings_mask_real = np.empty((len_dataset, 2048)) # conterr√† gli embeddings dell'inception per le maschere dei reali
    vettore_embeddings_mask_fake = np.empty((len_dataset, 2048))
    cnt_embeddings = 0

    # SSIM SCORE
    ssim_scores = np.empty(len_dataset)
    mask_ssim_scores = np.empty(len_dataset)

    # LOSS SCORE
    loss_scores = np.empty(len_dataset)

    # Predizione
    for cnt_img in range(len_dataset):
        sys.stdout.write(f"\rProcessamento immagine {cnt_img + 1} / {len_dataset}")
        sys.stdout.flush()
        batch = next(dataset)
        Ic = batch[0]  # [batch, 96, 128, 1]
        It = batch[1]  # [batch, 96,128, 1]
        Pt = batch[2]  # [batch, 96,128, 14]
        Mt = batch[3]  # [batch, 96,128, 1]
        Mc = batch[4]  # [batch, 96,128, 1]

        mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
        mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))
        mask_It = It * Mt

        # Predizione G1
        pred_generated, mask_pred_generated = None, None
        if len(models) == 1:
            pred_generated = G1.prediction(Ic, Pt)
            mask_pred_generated = pred_generated * Mt
        elif len(models) == 2:
            I_PT1 = G1.prediction(Ic, Pt)
            I_D = G2.prediction(I_PT1, Ic)
            I_D = tf.cast(I_D, dtype=tf.float16)
            pred_generated = I_PT1 + I_D  # [batch, 96, 128, 1]
            mask_pred_generated = pred_generated * Mt

        # Preprocesso immagini per Inception model
        input_inception_real[cnt_img % batch_size] = _inception_preprocess_image(It, mean_1, unprocess_function=dataset_module.unprocess_image)
        input_inception_fake[cnt_img % batch_size] = _inception_preprocess_image(pred_generated, mean_0, unprocess_function=dataset_module.unprocess_image)
        input_inception_mask_real[cnt_img % batch_size] = _inception_preprocess_image(mask_It, mean_1, unprocess_function=dataset_module.unprocess_image)
        input_inception_mask_fake[cnt_img % batch_size] = _inception_preprocess_image(mask_pred_generated, mean_0, unprocess_function=dataset_module.unprocess_image)

        # Computazinoe embeddings
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

        # Calcolo metriche
        ssim_scores[cnt_img] = G1.ssim(pred_generated, It, mean_0, mean_1, unprocess_function=dataset_module.unprocess_image)
        mask_ssim_scores[cnt_img] = G1.mask_ssim(pred_generated, It, Mt, mean_0, mean_1, unprocess_function=dataset_module.unprocess_image)
        loss_scores[cnt_img] = G1.PoseMaskloss(pred_generated, It, Mt)

    del batch

    # Salvataggio embeddings
    np.save(os.path.join(path_embeddings, "real_2048_embedding.npy"), vettore_embeddings_real)
    np.save(os.path.join(path_embeddings, "fake_2048_embedding.npy"), vettore_embeddings_fake)
    np.save(os.path.join(path_embeddings, "mask_real_2048_embedding.npy"), vettore_embeddings_mask_real)
    np.save(os.path.join(path_embeddings, "mask_fake_2048_embedding.npy"), vettore_embeddings_mask_fake)

    # FID
    fid_score = _calculate_FID_score(vettore_embeddings_real, vettore_embeddings_fake)
    mask_fid_score = _calculate_FID_score(vettore_embeddings_mask_real, vettore_embeddings_mask_fake)

    # IS
    inception_model = tf.keras.applications.InceptionV3(include_top=True, weights="imagenet", pooling='avg',
                                                        input_shape=(299, 299, 3))
    inputs = Input([2048])
    dense_layer = inception_model.layers[-1]
    dense_layer.set_weights(inception_model.layers[-1].get_weights())
    outputs = dense_layer(inputs)
    model_to_is = Model(inputs=inputs, outputs=outputs)

    del inception_model
    is_score = _calculate_IS_score(model_to_is, vettore_embeddings_fake)
    is_score_real = _calculate_IS_score(model_to_is, vettore_embeddings_real)

    mask_is_score = _calculate_IS_score(model_to_is, vettore_embeddings_mask_fake)
    mask_is_score_real = _calculate_IS_score(model_to_is, vettore_embeddings_mask_real)

    file = open(os.path.join(path_evaluation, "Scores.txt"), "w")
    text = "\nLOSS: {loss_value} " \
           "\nSSIM: {ssim_value} " \
           "\nFID: {fid_value} " \
           "\nIS: {is_value} " \
           "\nIS_real: {is_value_real}" \
           "\n" \
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
