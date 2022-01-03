import os
import numpy as np
import pickle

class Config:
    def __init__(self) :

        self.MODE = "train"  # ['train', 'test']
        self.DATASET = "Syntetich_complete"
        self.DATASET_type = "negative_no_flip_camp_5_keypoints_2_mask_1"
        self.ARCHITETURE = "mono"

        self._load_path()
        self._load_dataset_info()


    def _load_path(self):
        # - Path
        self.ROOT = '.'
        self.data_dir_path = os.path.join(self.ROOT, "data", self.DATASET)  #
        self.weigths_dir_path = os.path.join(self.ROOT, "weights")  # dove salvare i pesi
        self.logs_dir_path = os.path.join(self.ROOT, "logs")  # dove salvare i logs
        self.models_dir_path = os.path.join(self.ROOT, "models", self.ARCHITETURE)  # dove è presente il modello
        self.dataset_module_dir_path = os.path.join(self.ROOT, "datasets")  # dov è presente il modulo per processare il dataset
        self.data_tfrecord_path = os.path.join(self.data_dir_path, "tfrecord", self.DATASET_type)  # dove si trova il dataset in tfrecord

    def _load_dataset_info(self):
        # - Dataset

        self.img_H = 96  # 'input image height'
        self.img_W = 128  # 'input image width'
        if os.path.exists(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl')):
            with open(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl'), 'rb') as f:
                dic = pickle.load(f)

            self.name_tfrecord_train = os.path.join(self.data_tfrecord_path,
                                                    dic['train']['name_file'])  # nome dataset train
            self.name_tfrecord_valid = os.path.join(self.data_tfrecord_path,
                                                    dic['valid']['name_file'])
            self.name_tfrecord_test = os.path.join(self.data_tfrecord_path,
                                                   dic['test']['name_file'])

            self.dataset_train_len = int(dic['train']['tot'])  # numero di pair nel train
            self.dataset_valid_len = int(dic['valid']['tot'])
            self.dataset_test_len = int(dic['test']['tot'])

            self.dataset_train_list = dic['train']['list_pz']  # pz presenti nel train
            self.dataset_valid_list = dic['valid']['list_pz']
            self.dataset_test_list = dic['test']['list_pz']

        else:
            print("Dataset non presente. Eventualmente è ancora da formare")


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

        log_trainig = os.path.join(self.weigths_dir_path, 'dic.pkl')
        f = open(log_trainig, "wb")
        pickle.dump(dic, f)
        f.close()


