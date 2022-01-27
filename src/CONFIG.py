import os
import numpy as np
import pickle
from datetime import datetime

class Config:
    def __init__(self):
        self._load_enviroment_variable()
        self._load_path()
        self._load_dataset_info()
        self._load_G1_info()
        self._load_GAN_info()

    def _load_enviroment_variable(self):

        self.MODE = "inference_G1"  # ['train_G1', 'train_cDCGAN' 'inference_G1', 'inference_G2' 'evaluate']
        self.DATASET = "Syntetich_complete" # <nome_dataset>_[..]_[..]
        self.DATASET_type = "negative_no_flip_camp_5_keypoints_2_mask_1"
        self.ARCHITETURE = "bibranch"
        self.OUTPUTS_DIR = "output_mono" # directory in cui salvare tutti gli output durante il training

        self.G1_NAME_WEIGHTS_FILE = '../weights/Model_G1_Bibranch_epoch_005-loss_0.000-ssim_0.943-mask_ssim_0.984-val_loss_0.001-val_ssim_0.917-val_mask_ssim_0.979.hdf5'
        self.G2_NAME_WEIGHTS_FILE = '../weights/Model_G2_Bibranch_epoch_184-loss_0.69-ssmi_0.93-mask_ssmi_1.00-r_r_5499-im_0_5484-im_1_5464-val_loss_0.70-val_ssim_0.77-val_mask_ssim_0.98-val_r_r_400-val_im_0_400-val_im_1_400.hdf5'

    def _load_path(self):
        # - Path
        self.ROOT = '..'
        self.SRC = '.'
        self.OUTPUTS_DIR = os.path.join(self.ROOT, self.OUTPUTS_DIR)

        self.data_dir_path = os.path.join(self.ROOT, "data", self.DATASET)
        self.data_tfrecord_path = os.path.join(self.data_dir_path, "tfrecord", self.DATASET_type)  # dove si trova il dataset in tfrecord

        self.models_dir_path = os.path.join(self.SRC, "models", self.ARCHITETURE)  # dove sono presenti le architetture
        self.dataset_module_dir_path = os.path.join(self.SRC, "datasets")  # dov è presente il modulo per processare il dataset

        # check path
        #-ROOT
        assert os.path.exists(self.data_dir_path)
        assert os.path.exists(self.data_tfrecord_path)
        #-SRC
        assert os.path.exists(self.models_dir_path)
        assert os.path.exists(self.dataset_module_dir_path)
        assert os.path.exists(os.path.join(self.dataset_module_dir_path, self.DATASET.split('_')[0] + ".py"))
        #-OUTPUTS
        # TODO: milgiorare
        os.makedirs(self.OUTPUTS_DIR, exist_ok=False)

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

    def _load_G1_info(self):
        self.G1_epochs = 100
        self.G1_batch_size_train = 16  # grandezza del batch_size
        self.G1_batch_size_valid = 16  # grandezza del batch_size
        self.G1_lr_update_epoch = 1
        self.G1_drop_rate = 0.5
        self.G1_save_grid_ssim_epoch_train = 1
        self.G1_save_grid_ssim_epoch_valid = 1

    def _load_GAN_info(self):
        self.GAN_epochs = 200
        self.GAN_batch_size_train = 16  # grandezza del batch_size
        self.GAN_batch_size_valid = 16
        self.GAN_lr_update_epoch = 1000
        self.GAN_G2_drop_rate = 0.5
        self.GAN_D_drop_rate_D = 0.5
        self.GAN_save_grid_ssim_epoch_train = 1
        self.GAN_save_grid_ssim_epoch_valid = 1

    """
    Salvataggio configurazione utilizzate per il training
    """
    def save_config(self):
        list = self.__dict__
        name_file = os.path.join(self.OUTPUTS_DIR, "config_"+datetime.now().strftime('%d_%m_%Y_%H,%M')+".npy")
        np.save(name_file, list)




