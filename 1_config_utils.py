import os
import numpy as np
import pickle

class Config:
    def __init__(self) :

        self.Dataset = "Syntetich"
        self.type = "negative_syntetich"   # radius_<num> or new

        #Path
        self.data_path = './data/' + self.Dataset # dove si trova il dataset
        self.data_tfrecord_path = './data/' + self.Dataset + '/tfrecord/' + self.type  # dove si trova il dataset in tfrecord
        self.weigths_path = './weights' # dove salvare i pesi
        self.logs_path = './logs'
        self.training_weights_path = './Training/'

        #Dataset
        self.img_H = 96 #'input image height'
        self.img_W = 128 #'input image width'
        self.mean_img = 900
        if os.path.exists(os.path.join(self.data_tfrecord_path , 'pair_tot_sets.pkl')):
            with open(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl'), 'rb') as f:
                dic = pickle.load(f)

            self.name_tfrecord_train = dic['train']['name_file'] # nome dataset train
            self.name_tfrecord_valid = dic['valid']['name_file'] # nome dataset valid
            self.name_tfrecord_test = dic['test']['name_file']  # nome dataset test

            self.dataset_train_len = int(dic['train']['tot']) # numero di pair nel train
            self.dataset_valid_len = int(dic['valid']['tot'])  # numero di pair nel valid
            self.dataset_test_len = int(dic['test']['tot'])   # numero di pair nel test

            print("Lunghezza Sets:")
            print("- "+self.name_tfrecord_train+" : ", self.dataset_train_len )
            print("- "+self.name_tfrecord_valid+" : ", self.dataset_valid_len )
            print("- "+self.name_tfrecord_test+" : ", self.dataset_test_len )

            print("List pz:")
            print(dic['train']['list_pz'])
            print(dic['valid']['list_pz'])
            print(dic['test']['list_pz'])

        else:
            print("Dataset no presente. Eventualmente è ancora da formare")

        #model G1 / G2
        self.input_shape_g1 = [96, 128, 15]  # concat tra image_raw_0 a 1 channel e la posa a 14 channel
        self.input_shape_g2 = [96, 128, 2]  # concat tra image_raw_0 a 1 channel e l' output del generatore G1 a 1 canale
        self.input_shape_d = [96, 128, 1]

        # numero di blocchi residuali del G1. --> 4 con height 96
        # Per il G2 verrà considerato un repeat_num - 2
        self.repeat_num = int(np.log2(self.img_H)) - 2
        self.conv_hidden_num = 128 # numero di filtri del primo layer convoluzionale
        self.z_num = 64 # numero neuroni del fully connected del G1
        self.input_image_raw_channel = 1  # indica per le image_raw_0 il 1 GRAY, mi serve per la regressione di output della rete
        self.activation_fn = 'relu'
        self.min_fea_map_H = 12
        self.min_fea_map_W = 16
        self.keypoint_num = 14  # numero di keypoints

        # Training / test parameters
        self.epochs = 200
        self.batch_size_train = 16  # grandezza del batch_size
        self.batch_size_valid = 16  # grandezza del batch_size
        self.save_grid_ssim_epoch_valid = 1  # ogni quante epoche devo salvare la griglia per visualizzare le immagini predette dal G2
        self.save_grid_ssim_epoch_train = 1
        self.lr_update_epoch = 5  # epoche di aggiornameto del learning rate

        #google colab
        self.run_google_colab = False
        self.download_weight = 2 # step di epoche in cui andremo ad aggiornare il rar dei pesi

        self.data_format = 'channels_last'
