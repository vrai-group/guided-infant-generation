"""
Questo codice calcola:
1. embedded features di vgg16 sulla distribuzione delle immagini reali --> 512 features per immagine
2. calcola la PCA sulle features al punto sopra. Da 512 embedding features abbiamo un passaggio a 50 features per immagine
3. Andiamo a calcolare il TSNE sulle features del punto 2. Da 50 features passiamo a 2 features per la rappresentazione sul piano cartesiano
"""
import os
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

from utils.evaluation.tsne.plot_tsne import _plot

def _vgg_preprocess_image(image):

    image = tf.concat([image, image, image], axis=-1)
    image = cv2.resize(image.numpy()[0], (224, 224))
    image = vgg16.preprocess_input(image)
    image = np.reshape(image, (1, 224, 224, 3))

    return image

def _get_vgg_model():
    vgg_model = VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
    layer = vgg_model.get_layer(name="global_average_pooling2d")
    feature_extractor = Model(inputs=vgg_model.inputs, outputs=layer.output)

    return feature_extractor


#############################
# REAL
#############################
"""
Estrae le features da VGG16
"""
def _extract_features_vgg_real(list_sets, dataset_module, dict_data, feature_extractor):


    for dataset in list_sets:
        name_dataset, dataset_len = dataset[0], dataset[1]
        type_dataset = name_dataset.split(os.path.sep)[-1].split('.')[0].split('_')[1]  # ['train', 'valid', 'test']

        # Dataset
        dataset = dataset_module.get_unprocess_dataset(name_tfrecord=name_dataset)
        dataset = dataset_module.preprocess_dataset(dataset)
        dataset = dataset.batch(1)
        dataset = iter(dataset)
        print("\n")
        for cnt_img in range(dataset_len):
            sys.stdout.write(f"\r-Processamento {type_dataset} immagine {cnt_img + 1} / {dataset_len}")
            sys.stdout.flush()
            batch = next(dataset)
            It = batch[1]  # [batch, 96,128, 1]
            pz_1 = batch[6]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            mean_1 = tf.reshape(batch[10], (-1, 1, 1, 1))
            pz_1 = pz_1.numpy()[0].decode('utf-8')
            name_1 = name_1.numpy()[0].decode('utf-8')

            # Process for VGG16
            image_raw_1_unp = tf.cast(dataset_module.unprocess_image(It, mean_1), dtype=tf.uint8)
            image_process = _vgg_preprocess_image(image_raw_1_unp)

            features_real = feature_extractor.predict(image_process)[0] #[batch, 512]

            key = type_dataset + '_' + str(cnt_img)
            dict_data[key] = {}
            dict_data[key]['path_target'] = pz_1 + '/' + name_1  # path della target o meglio della reale
            dict_data[key]['features_vgg_real'] = features_real

    return dict_data

def _obtain_pca_real(dict_data):
    vgg_features_real = np.array([dict_data[k]['features_vgg_real'] for k, v in dict_data.items()]) # creo un numpy array con le fetures
    standardize_features = StandardScaler().fit_transform(vgg_features_real)  # Standardizzazione
    pca_features = PCA(n_components=50, svd_solver='auto').fit_transform(standardize_features)  # PCA

    for i, elem in enumerate(dict_data.items()):
        key, _ = elem
        dict_data[key]['features_pca_real'] = pca_features[i]

    return dict_data

def _obtain_tsne_real(dict_data, perplexity):
    pca_features_real = np.array([dict_data[k]['features_pca_real'] for k, v in dict_data.items()])
    tsne_features = TSNE(n_components=2, perplexity=perplexity, n_iter=6000).fit_transform(pca_features_real)  # TSNE

    # Normalizzo min-max
    tx, ty = tsne_features[:, 0], tsne_features[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    for i, elem in enumerate(dict_data.items()):
        key, _ = elem
        dict_data[key]['features_tsne_real_'+str(perplexity)] = np.array([tx[i],ty[i]])

    return dict_data

def _start_real(list_sets, list_perplexity, dataset_module, feature_extractor):
    print("\nCalcolo del tsne sulle real")

    dict_data_real = {}
    dict_data_real = _extract_features_vgg_real(list_sets, dataset_module, dict_data_real, feature_extractor)
    dict_data_real = _obtain_pca_real(dict_data_real)

    for perplexity in list_perplexity:
        print("\n- Perplexity: ", str(perplexity))
        dict_data_real = _obtain_tsne_real(dict_data_real, perplexity)

    return dict_data_real

##############################
# GENERATED
##############################
def _extract_features_vgg_generated(G1, G2, list_sets, dataset_module, dict_data, feature_extractor):

    for dataset in list_sets:
        name_dataset, dataset_len = dataset[0], dataset[1]
        type_dataset = name_dataset.split(os.path.sep)[-1].split('.')[0].split('_')[1]  # ['train', 'valid', 'test']

        # Dataset
        dataset = dataset_module.get_unprocess_dataset(name_dataset)
        dataset = dataset_module.preprocess_dataset(dataset)
        dataset = dataset.batch(1)
        dataset = iter(dataset)
        print("\n")
        for cnt_img in range(dataset_len):
            sys.stdout.write(f"\rProcessamento {type_dataset} immagine {cnt_img + 1} / {dataset_len}")
            sys.stdout.flush()
            batch = next(dataset)
            Ic = batch[0]  # [batch, 96, 128, 1]
            pose_1 = batch[2]  # [batch, 96,128, 14]
            mean_0 = tf.reshape(batch[9], (-1, 1, 1, 1))
            pz_1 = batch[6]  # [batch, 1]
            name_1 = batch[8]  # [batch, 1]
            pz_1 = pz_1.numpy()[0].decode('utf-8')
            name_1 = name_1.numpy()[0].decode('utf-8')

            # Predizione
            I_PT1 = G1.prediction(Ic, pose_1)
            I_D = G2.prediction(I_PT1, Ic)
            I_PT2 = I_PT1 + I_D

            # Unprocess for VGG16
            predizione_unp = tf.cast(dataset_module.unprocess_image(I_PT2, mean_0), dtype=tf.uint8)
            image_process = _vgg_preprocess_image(predizione_unp)

            features_generated = feature_extractor.predict(image_process)[0]

            key = type_dataset + '_' + str(cnt_img)
            dict_data[key] = {}
            dict_data[key]['path_target'] = pz_1 + '/' + name_1  # path della target o meglio della reale
            dict_data[key]['features_vgg_generated'] = features_generated

    return dict_data

def _obtain_pca_generated(dict_data):
    vgg_features_real = np.array([dict_data[k]['features_vgg_generated'] for k, v in dict_data.items()])
    standardize_features = StandardScaler().fit_transform(vgg_features_real)  # Standardizzazione
    pca_features = PCA(n_components=50, svd_solver='auto').fit_transform(standardize_features)  # PCA

    for i, elem in enumerate(dict_data.items()):
        key, _ = elem
        dict_data[key]['features_pca_generated'] = pca_features[i]

    return dict_data

def _obtain_tsne_generated(dict_data, perplexity):
    pca_features_real = np.array([dict_data[k]['features_pca_generated'] for k, v in dict_data.items()])
    tsne_features = TSNE(n_components=2, perplexity=perplexity, n_iter=6000).fit_transform(
        pca_features_real)  # TSNE

    # Normalizzo min-max
    tx, ty = tsne_features[:, 0], tsne_features[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    for i, elem in enumerate(dict_data.items()):
        key, _ = elem
        dict_data[key]['features_tsne_generated_'+str(perplexity)] = np.array([tx[i],ty[i]])

    return dict_data

def _start_generated(list_sets, list_perplexity, G1, G2, dataset_module, feature_extractor):
    print("\nCalcolo del tsne sulle generated")

    dict_data_generated = {}
    dict_data_generated = _extract_features_vgg_generated(G1, G2, list_sets, dataset_module, dict_data_generated, feature_extractor)
    dict_data_generated = _obtain_pca_generated(dict_data_generated)

    for perplexity in list_perplexity:
        print("\n- Perplexity: ", str(perplexity))
        dict_data_generated = _obtain_tsne_generated(dict_data_generated, perplexity)

    return dict_data_generated

def start(list_sets, list_perplexity, G1, G2, dataset_module, dir_to_save, key_image_interested, save_fig_plot=True, save_box_plot=True):
    # Extractor
    feature_extractor = _get_vgg_model()

    dict_data_real = _start_real(list_sets, list_perplexity, dataset_module, feature_extractor)
    dict_data_generated = _start_generated(list_sets, list_perplexity, G1, G2, dataset_module, feature_extractor)

    print("\n- Unisco i dizionari")
    dict_features_tot = {}
    for key_principal in list(dict_data_real.keys()):
        dict_features_tot[key_principal] = {}

        # Reali
        for key_real in dict_data_real[key_principal].keys():
            dict_features_tot[key_principal][key_real] = dict_data_real[key_principal][key_real]
        # Generate
        for key_generated in dict_data_generated[key_principal].keys():
            dict_features_tot[key_principal][key_generated] = dict_data_generated[key_principal][key_generated]

    # Salvataggio del file
    name_file = os.path.join(dir_to_save, "dict_vgg_pca_tsne_features_real_and_generated.npy")
    print("\n- Salvo il dict contenetente le featuress. Nome: ", name_file)
    np.save(name_file, dict)

    if save_fig_plot:
        print("-\n Plotto i grafici. La Key_image_interested ??: "+str(key_image_interested))
        _plot(dict_features_tot, list_perplexity, dir_to_save, key_image_interested=key_image_interested)