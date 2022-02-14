import os
import numpy as np
import pickle
from datetime import datetime
import shutil

class Config:
    def __init__(self):
        self.__load_enviroment_variable()
        self.__load_general_path()
        self.__load_dataset_info()
        self.__load_G1_info()
        self.__load_GAN_info()

    def __load_enviroment_variable(self):
        self.MODE = "tsne_GAN"  # ['train_G1', 'train_cDCGAN', 'evaluate_G1', 'evaluate_GAN', 'tsne_GAN', 'inference_G1', 'inference_G2]
        self.DATA = "Syntetich_complete" # [tipologia][underscore][note]
        self.DATASET_type = self.DATA.split('_')[0]
        self.DATASET = "dataset_di_testing"
        self.ARCHITETURE = "bibranch"
        self.OUTPUTS_DIR = "output_bibranch" # nome directory in cui salvare tutti gli output durante il training

        self.G1_NAME_WEIGHTS_FILE = 'Model_G1_Bibranch_epoch_005-loss_0.000.hdf5'
        self.G2_NAME_WEIGHTS_FILE = 'Model_G2_Bibranch_epoch_184-loss_0.69.hdf5'

    def __load_general_path(self):
        self.ROOT = '..'
        self.SRC = '.'
        self.OUTPUTS_DIR = os.path.join(self.ROOT, self.OUTPUTS_DIR)

        self.data_dir_path = os.path.join(self.ROOT, "data", self.DATA)
        self.data_tfrecord_path = os.path.join(self.data_dir_path, "tfrecord", self.DATASET)  # dove si trova il dataset in tfrecord
        self.models_dir_path = os.path.join(self.SRC, "models", self.ARCHITETURE)  # dove sono presenti le architetture
        self.dataset_module_dir_path = os.path.join(self.SRC, "datasets")  # dov è presente il modulo per processare il dataset

        # check path
        #-ROOT
        assert os.path.exists(self.data_dir_path)
        assert os.path.exists(self.data_tfrecord_path)
        #-SRC
        assert os.path.exists(self.models_dir_path)
        assert os.path.exists(self.dataset_module_dir_path)
        assert os.path.exists(os.path.join(self.dataset_module_dir_path, self.DATA.split('_')[0] + ".py"))
        #-OUTPUTS
        if os.path.exists(self.OUTPUTS_DIR):
            r_v = input("La cartella di output esiste già. Sovrascriverla? Questo comporterà la perdita di tutti i dati Yes[Y] No[N]")
            assert r_v == "Y" or r_v == "N" or r_v == "y" or r_v == "n"
            if r_v == "Y" or r_v == "y":
                shutil.rmtree(self.OUTPUTS_DIR)
        if not os.path.exists(self.OUTPUTS_DIR):
            os.mkdir(self.OUTPUTS_DIR)

    def load_train_path_G1(self):
        self.G1_logs_dir_path = os.path.join(self.OUTPUTS_DIR, "logs", "G1")
        self.G1_weights_path = os.path.join(self.OUTPUTS_DIR, "weights", "G1")
        self.G1_grid_path = os.path.join(self.OUTPUTS_DIR, "grid", "G1")

        os.makedirs(self.G1_logs_dir_path, exist_ok=True)
        os.makedirs(self.G1_weights_path, exist_ok=True)
        os.makedirs(self.G1_grid_path, exist_ok=True)

    def load_inference_path_G1(self):
        self.G1_name_dir_test_inference = os.path.join(self.OUTPUTS_DIR, "inference_test_set","G1")
        os.makedirs(self.G1_name_dir_test_inference, exist_ok=True)

    def load_evaluate_path_G1(self):
        self.G1_evaluation_path = os.path.join(self.OUTPUTS_DIR, "evaluation","G1")
        os.makedirs(self.G1_evaluation_path, exist_ok=True)

    def load_train_path_GAN(self):
        self.GAN_logs_dir_path = os.path.join(self.OUTPUTS_DIR, "logs", "GAN")
        self.GAN_weights_path = os.path.join(self.OUTPUTS_DIR, "weights", "GAN")
        self.GAN_grid_path = os.path.join(self.OUTPUTS_DIR, "grid", "GAN")

        os.makedirs(self.GAN_logs_dir_path, exist_ok=True)
        os.makedirs(self.GAN_weights_path, exist_ok=True)
        os.makedirs(self.GAN_grid_path, exist_ok=True)

    def load_inference_path_GAN(self):
        self.GAN_name_dir_test_inference = os.path.join(self.OUTPUTS_DIR, "inference_test_set","GAN")
        os.makedirs(self.GAN_name_dir_test_inference, exist_ok=True)

    def load_evaluate_path_GAN(self):
        self.GAN_evaluation_path = os.path.join(self.OUTPUTS_DIR, "evaluation","GAN")
        os.makedirs(self.GAN_evaluation_path, exist_ok=True)

    def __load_dataset_info(self):
        # - Dataset
        self.img_H = 96  # 'input image height'
        self.img_W = 128  # 'input image width'
        # Se l assert va in errore il dataset non è presente
        assert os.path.exists(os.path.join(self.data_tfrecord_path, 'sets_config.pkl')) == True

        with open(os.path.join(self.data_tfrecord_path, 'sets_config.pkl'), 'rb') as f:
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

    def __load_G1_info(self):
        self.G1_epochs = 100
        self.G1_batch_size_train = 16  # grandezza del batch_size
        self.G1_batch_size_valid = 16  # grandezza del batch_size
        self.G1_lr_update_epoch = 1
        self.G1_drop_rate = 0.5
        self.G1_save_grid_ssim_epoch_train = 1
        self.G1_save_grid_ssim_epoch_valid = 1

    def __load_GAN_info(self):
        self.GAN_epochs = 200
        self.GAN_batch_size_train = 16  # grandezza del batch_size
        self.GAN_batch_size_valid = 16
        self.GAN_lr_update_epoch = 1000
        self.GAN_G2_drop_rate = 0.5
        self.GAN_D_drop_rate_D = 0.5
        self.GAN_save_grid_ssim_epoch_train = 1
        self.GAN_save_grid_ssim_epoch_valid = 1