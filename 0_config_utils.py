import os
import numpy as np
import pickle

class Config:
    def __init__(self) :

        self.Dataset = "BabyPose"
        self.radius = "10"

        #Path
        self.data_path = './data/' + self.Dataset # dove si trova il dataset
        self.data_tfrecord_path = './data/' + '/tfrecord/' + "radius_" + self.radius  # dove si trova il dataset in tfrecord
        self.data_annotations_path = './data/annotations' # dove si trovano le annotazioni
        self.logs_cancellate_path = './masks/logs_cancellazione' # dove si trovano i file npy contenente i nomi delle immagini da non considerare poichè manca uno di questi Keypoint: 3,5,10,11
        self.weigths_path = './weights' # dove salvare i pesi
        self.logs_path = './logs'
        self.training_weights_path = './training/'

        #Dataset
        self.img_H = 96 #'input image height'
        self.img_W = 128 #'input image width'
        if self.Dataset == "BabyPose" and os.path.exists(os.path.join(self.data_tfrecord_path , 'pair_tot_sets.pkl')):
            with open(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl'), 'rb') as f:
                dic = pickle.load(f)
            self.dataset_train_len = int(dic['train']['tot']) # numero di pair nel train
            self.dataset_valid_len = int(dic['valid']['tot'])  # numero di pair nel valid
            self.dataset_test_len = int(dic['test']['tot'])   # numero di pair nel test
            print("Lunghezza Sets:")
            print("- "+dic['train']['name_file']+" : ", self.dataset_train_len )
            print("- "+dic['valid']['name_file']+" : ", self.dataset_valid_len )
            print("- "+dic['test']['name_file']+" : ", self.dataset_test_len )
        else:
            print("Dataset no presente. Eventualmente è ancora da formare")

        #model G1 / G2
        self.input_shape_g1 = [96, 128, 17]  # concat tra image_raw_0 a 1 channel e la posa a 14 channel
        self.input_shape_g2 = [96, 128, 2]  # concat tra image_raw_0 a 1 channel e l' output del generatore G1 a 1 canale

        # numero di blocchi residuali del G1. --> 4 con height 96
        # Per il G2 verrà considerato un repeat_num - 2
        self.repeat_num = int(np.log2(self.img_H)) - 2
        self.conv_hidden_num = 128 # numero di filtri del primo layer convoluzionale. dall helper n in the paper
        self.z_num = 64 # numero neuroni del fully connected del G1
        self.input_image_raw_channel = 3  # indica per le image_raw_0 il 1 GRAY, mi serve per la regressione di output della rete
        self.activation_fn = 'relu'
        self.min_fea_map_H = 12
        self.min_fea_map_W = 16
        self.keypoint_num = 14  # numero di mappe

        # Training / test parameters
        self.is_train = False
        self.epochs = 100
        self.batch_size_train = 16  # grandezza del batch_size
        self.batch_size_valid = 16  # grandezza del batch_size
        self.save_grid_ssim_epoch = 1  # ogni quante epoche devo salvare la griglia per visualizzare le immagini predette dal G2
        self.lr_update_epoch = 10  # epoche di aggiornameto del learning rate
        #google colab
        self.run_google_colab = True
        self.download_weight = 5 # step di epoche in cui andremo ad aggiornare il rar dei pesi
