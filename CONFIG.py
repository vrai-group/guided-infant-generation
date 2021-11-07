import os
import numpy as np
import pickle

class Config:
    def __init__(self) :

        self.DATASET = "Syntetich_complete"  # tag_note
        self.DATASET_type = "negative_no_flip_camp_5_keypoints_2_mask_1"

        self.ARCHITECTURE = "mono"  # il nome deve match la cartella in models/Syntetich
        self.MODE = "train" #['train', 'test']

        # - Path
        self.ROOT = '.'
        self.data_path = os.path.join(self.ROOT, "data", self.DATASET) # dove si trova il dataset
        self.data_tfrecord_path = os.path.join(self.data_path, "tfrecord", self.DATASET_type)  # dove si trova il dataset in tfrecord
        self.weigths_path = os.path.join(self.ROOT, "weights")  # dove salvare i pesi
        self.logs_path = os.path.join(self.ROOT, "logs")
        self.dataset_file_path = os.path.join(self.ROOT, "dataset")
        self.models_path = os.path.join(self.ROOT, "models", self.ARCHITECTURE)

        # - Dataset
        self.img_H = 96 #'input image height'
        self.img_W = 128 #'input image width'
        if os.path.exists(os.path.join(self.data_tfrecord_path , 'pair_tot_sets.pkl')):
            with open(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl'), 'rb') as f:
                dic = pickle.load(f)

            self.name_tfrecord_train = dic['train']['name_file'] # nome dataset train
            self.name_tfrecord_valid = dic['valid']['name_file'] # nome dataset valid
            self.name_tfrecord_test = dic['test']['name_file']  # nome dataset test

            self.dataset_train_len = int(dic['train']['tot']) # numero di pair nel train
            self.dataset_valid_len = int(dic['valid']['tot'])  # numero di pair nel valid
            self.dataset_test_len = int(dic['test']['tot'])   # numero di pair nel test

            self.dataset_train_list = dic['train']['list_pz']
            self.dataset_valid_list = dic['valid']['list_pz']
            self.dataset_test_list = dic['test']['list_pz']

        else:
            print("Dataset no presente. Eventualmente è ancora da formare")


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

        # -G1
        self.trainig_G1 = True
        # -- Model

        # -- Training / test parameters
        self.epochs_G1 = 100
        self.lr_update_epoch_G1 = 1
        self.lr_initial_G1 = 2e-5
        self.drop_rate_G1 = 0.5

        # -GAN
        self.trainig_GAN = False
        # -- Model
        self.input_shape_G2 = [96, 128, 2]  # concat tra image_raw_0 a 1 channel e l' output del generatore G1 a 1 canale
        self.input_shape_D = [96, 128, 1]

        # -- Training / test parameters
        self.epochs_GAN = 200
        self.lr_update_epoch_GAN = 1000
        self.lr_initial_G2 = 2e-5
        self.lr_initial_D = 2e-5
        self.drop_rate_GAN = self.drop_rate_G1

        # - General (Sia per G1 che per GAN)
        self.save_grid_ssim_epoch_valid = 1
        self.save_grid_ssim_epoch_train = 1

        self.batch_size_train = 16  # grandezza del batch_size
        self.batch_size_valid = 16  # grandezza del batch_size

        self.run_google_colab = False
        self.download_weight = 1 # step di epoche in cui andremo ad aggiornare il rar dei pesi

        self.data_format = 'channels_last'


    def print_info(self):

        print("Lunghezza Sets:")
        print("- " + self.name_tfrecord_train + " : ", self.dataset_train_len)
        print("- " + self.name_tfrecord_valid + " : ", self.dataset_valid_len)
        print("- " + self.name_tfrecord_test + " : ", self.dataset_test_len)

        print("List pz:")
        print(self.dataset_train_list)
        print(self.dataset_valid_list)
        print(self.dataset_test_list )

        if self.trainig_G1:
            print("Allenamento G1:")
            print("Epoche: ", self.epochs_G1)
            print("lr iniziale: ", self.lr_initial_G1)
            print("lr update rate: ",self.lr_update_epoch_G1," epoche")
            print("lr drop rate: ", self.drop_rate_G1)
            print("Num batch train: ", self.batch_size_train)
            print("Num batch valid: ", self.batch_size_train)

        if self.trainig_GAN:
            print("Allenamento GAN:")
            print("Epoche: ", self.epochs_GAN)
            print("lr iniziale G2: ", self.lr_initial_G2)
            print("lr iniziale D: ", self.lr_initial_D)
            print("lr update rate: ",self.lr_update_epoch_GAN," epoche")
            print("lr drop rate: ", self.drop_rate_GAN)
            print("Num batch train: ", self.batch_size_train)
            print("Num batch valid: ", self.batch_size_train)

    def save_info(self):

        dic_G1 = {

            "name_dataset": self.DATASET,
            "type_dataset": self.DATASET_type,
            "Allenamento G1": True,
            "lr iniziale": self.lr_initial_G1,
            "lr update rate epoche": self.lr_update_epoch_G1,
            "lr drop rate": self.drop_rate_G1,
            "Num batch train": self.batch_size_train,
            "Num batch valid": self.batch_size_train
        }

        dic_GAN = {

            "name_dataset": self.DATASET,
            "type_dataset": self.DATASET_type,
            "Allenamento GAN":True,
            "lr iniziale G2": self.lr_initial_G2,
            "lr iniziale D": self.lr_initial_D,
            "lr update rate": self.lr_update_epoch_GAN,
            "lr drop rate": self.drop_rate_GAN,
            "Num batch train": self.batch_size_train,
            "Num batch valid": self.batch_size_train

        }

        dic = None
        if self.trainig_G1:
            dic = dic_G1
        elif self.trainig_GAN:
            dic = dic_GAN

        log_trainig = os.path.join(self.weigths_path, 'dic.pkl')
        f = open(log_trainig, "wb")
        pickle.dump(dic, f)
        f.close()


