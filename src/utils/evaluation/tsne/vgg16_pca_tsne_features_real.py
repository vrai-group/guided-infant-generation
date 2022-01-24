"""
Questo codice calcola:
-features di vgg16 sulla distribuzione reali di tutt e tre i set --> 512 features per immagine
-calcola la pca sulle features al punto sopra  --> 50 features per immagine
"""
import sys
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import VGG16

def _vgg_preprocess_image(image):

    image = tf.concat([image, image, image], axis=-1)
    image = cv2.resize(image.numpy()[0], (224, 224))
    image = vgg16.preprocess_input(image)
    image = np.reshape(image, (1, 224, 224, 3))

    return image

def _extract_features_real(list_sets, dataset_module):
    dict_data = {}

    # Extractor
    vgg_model = VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
    layer = vgg_model.get_layer(name="global_average_pooling2d")
    feature_extractor = Model(inputs=vgg_model.inputs, outputs=layer.output)

    for dataset in list_sets:
        name_dataset, dataset_len = dataset[0], dataset[1]
        type_dataset = name_dataset.split('_')[1].split('.tfrecord')[0]  # ['train', 'valid', 'test']

        # Dataset
        dataset = dataset_module.get_unprocess_dataset(name_tfrecord=name_dataset)
        dataset = dataset_module.get_preprocess_G1_dataset(dataset)
        dataset = dataset.batch(1)
        dataset = iter(dataset)

        for cnt_img in range(dataset_len):
            sys.stdout.write("\rProcessamento {type_dataset} immagine {cnt} / {tot}".format(cnt=cnt_img + 1,
                                                                                            tot=dataset_len,
                                                                                            type_dataset=type_dataset))
            sys.stdout.flush()
            batch = next(dataset)
            image_raw_1 = batch[1]  # [batch, 96,128, 1]
            pz_1 = batch[6]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))
            pz_1 = pz_1.numpy()[0].decode('utf-8')
            name_1 = name_1.numpy()[0].decode('utf-8')

            # Process for VGG16
            image_raw_1_unp = tf.cast(dataset_module.unprocess_image(image_raw_1, mean_1, 32765.5), dtype=tf.uint8)
            image_process = _vgg_preprocess_image(image_raw_1_unp)

            features_real = feature_extractor.predict(image_process)[0] #[batch, 512]

            key = type_dataset + '_' + str(cnt_img)
            dict_data[key] = {}
            dict_data[key]['path_target'] = pz_1 + '/' + name_1  # path della target o meglio della reale
            dict_data[key]['features_vgg_real'] = features_real

        sys.stdout.write("\n#######\n")
        sys.stdout.flush()

    return dict_data

def _obtain_pca_real(dict_data):
    vgg_features_real = np.array([dict_data[k]['features_vgg_real'] for k, v in dict_data.items()])
    standardize_features = StandardScaler().fit_transform(vgg_features_real)  # Standardizzazione
    pca_features = PCA(n_components=50, svd_solver='auto').fit_transform(standardize_features)  # PCA

    for i, elem in enumerate(dict_data.items()):
        key, _ = elem
        dict_data[key]['features_pca_real'] = pca_features[i]

    return dict_data

def _obtain_tsne_real(dict_data, perplexity):
    pca_features_real = np.array([dict_data[k]['features_pca_real'] for k, v in dict_data.items()])
    tsne_features = TSNE(n_components=2, perplexity=perplexity, n_iter=6000).fit_transform(
        pca_features_real)  # TSNE

    # Normalizzo min-max
    tx, ty = tsne_features[:, 0], tsne_features[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    for i, elem in enumerate(dict_data.items()):
        key, _ = elem
        dict_data[key]['features_tsne_real_'+str(perplexity)] = np.array([tx[i],ty[i]])

def start(list_sets, list_perplexity, dataset_module):

    dict_vgg_features_real = _extract_features_real(list_sets, dataset_module)
    dict_vgg_pca_features_real = _obtain_pca_real(dict_vgg_features_real)

    for perplexity in list_perplexity:
        _obtain_tsne_real(dict_vgg_pca_features_real,perplexity)

    #TODO definire il salvataggio
    #np.save("dict_vgg_pca_tsne_features_real.npy", dict_vgg_pca_features_real)