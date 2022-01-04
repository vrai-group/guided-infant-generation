import os
import numpy as np
import pickle


class Config:
    def __init__(self):
        self.MODE = "train"  # ['train', 'test']
        self.DATASET = "Syntetich_complete"
        self.DATASET_type = "negative_no_flip_camp_5_keypoints_2_mask_1"
        self.ARCHITETURE = "mono"

        self._load_path()
        self._load_dataset_info()
        self._load_train_info()

    def _load_path(self):
        # - Path
        self.ROOT = '.'
        self.data_dir_path = os.path.join(self.ROOT, "data", self.DATASET)
        self.weigths_dir_path = os.path.join(self.ROOT, "weights")  # dove salvare i pesi
        self.logs_dir_path = os.path.join(self.ROOT, "logs")  # dove salvare i logs
        self.models_dir_path = os.path.join(self.ROOT, "models", self.ARCHITETURE)  # dove è presente il modello
        self.dataset_module_dir_path = os.path.join(self.ROOT,
                                                    "datasets")  # dov è presente il modulo per processare il dataset
        self.data_tfrecord_path = os.path.join(self.data_dir_path, "tfrecord",
                                               self.DATASET_type)  # dove si trova il dataset in tfrecord

    def _load_dataset_info(self):
        # - Dataset
        self.img_H = 96  # 'input image height'
        self.img_W = 128  # 'input image width'
        # Se l assert va in errore il dataset non è presente
        assert os.path.exists(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl')) == True

        with open(os.path.join(self.data_tfrecord_path, 'pair_tot_sets.pkl'), 'rb') as f:
            dic = pickle.load(f)
            # nome set
            self.name_tfrecord_train = os.path.join(self.data_tfrecord_path, dic['train']['name_file'])
            self.name_tfrecord_valid = os.path.join(self.data_tfrecord_path, dic['valid']['name_file'])
            self.name_tfrecord_test = os.path.join(self.data_tfrecord_path, dic['test']['name_file'])
            # numero di pair
            self.dataset_train_len = int(dic['train']['tot'])
            self.dataset_valid_len = int(dic['valid']['tot'])
            self.dataset_test_len = int(dic['test']['tot'])
            # lista pz presenti
            self.dataset_train_list = dic['train']['list_pz']  # pz presenti nel train
            self.dataset_valid_list = dic['valid']['list_pz']
            self.dataset_test_list = dic['test']['list_pz']


    def _load_train_info(self):
        self.epochs_G1 = 100
        self.batch_size_train = 16  # grandezza del batch_size
        self.batch_size_valid = 16  # grandezza del batch_size
