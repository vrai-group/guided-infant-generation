import yaml
import os
import pickle
import shutil


class Config:
    def __init__(self):
        self.__load_configuration_file()
        self.__load_general_path()
        self.__load_dataset_info()

    def __load_configuration_file(self):
        configuration = yaml.load(stream=open("../configuration.yml", "r"), Loader=yaml.FullLoader)

        enviroment_variable = configuration['enviroment_variable']
        self.MODE = enviroment_variable['MODE']
        self.DATASET = enviroment_variable['DATASET']
        self.DATASET_CONFIGURATION = enviroment_variable['DATASET_CONFIGURATION']
        self.ARCHITETURE = enviroment_variable['ARCHITETURE']
        self.OUTPUTS_DIR = enviroment_variable['OUTPUTS_DIR']
        if self.MODE in ['inference_G1', 'evaluate_G1', 'train_cDCGAN', 'evaluate_GAN', 'tsne_GAN', 'inference_GAN']:
            self.G1_NAME_WEIGHTS_FILE = enviroment_variable['G1_NAME_WEIGHTS_FILE']
        if self.MODE in ['evaluate_GAN', 'inference_GAN', 'tsne_GAN']:
            self.G2_NAME_WEIGHTS_FILE = enviroment_variable['G2_NAME_WEIGHTS_FILE']

        if self.MODE in ['train_G1']:
            G1_train_info = configuration['G1_train_info']
            self.G1_epochs = G1_train_info["epochs"]
            self.G1_batch_size_train = G1_train_info["batch_size_train"]
            self.G1_batch_size_valid = G1_train_info["batch_size_valid"]
            self.G1_lr_update_epoch = G1_train_info["lr_update_epoch"]
            self.G1_drop_rate = G1_train_info["drop_rate"]
            self.G1_save_grid_ssim_epoch_train = G1_train_info["save_grid_ssim_epoch_train"]
            self.G1_save_grid_ssim_epoch_valid = G1_train_info["save_grid_ssim_epoch_valid"]

        if self.MODE in ['train_cDCGAN']:
            GAN_train_info = configuration['GAN_train_info']
            self.GAN_epochs = GAN_train_info["epochs"]
            self.GAN_batch_size_train = GAN_train_info["batch_size_train"]
            self.GAN_batch_size_valid = GAN_train_info["batch_size_valid"]
            self.GAN_lr_update_epoch = GAN_train_info["lr_update_epoch"]
            self.GAN_G2_drop_rate = GAN_train_info["G2_drop_rate"]
            self.GAN_D_drop_rate = GAN_train_info["D_drop_rate"]
            self.GAN_save_grid_ssim_epoch_train = GAN_train_info["save_grid_ssim_epoch_train"]
            self.GAN_save_grid_ssim_epoch_valid = GAN_train_info["save_grid_ssim_epoch_valid"]

    def __load_general_path(self):
        self.ROOT = '..'
        self.SRC = '.'
        self.DATASET_TYPE = self.DATASET.split('_')[0]  # Per la lettura del file di processamento
        self.OUTPUTS_DIR = os.path.join(self.ROOT, self.OUTPUTS_DIR)

        self.dataset_dir_path = os.path.join(self.ROOT, "data", self.DATASET)
        self.dataset_configuration_path = os.path.join(self.dataset_dir_path, "tfrecord",
                                                       self.DATASET_CONFIGURATION)  # dove si trova il dataset la configurazione del dataset
        self.dataset_module_dir_path = os.path.join(self.SRC,
                                                    "datasets")  # dov è presente il modulo per processare il dataset
        self.models_dir_path = os.path.join(self.SRC, "models", self.ARCHITETURE)  # dove sono presenti le architetture

        # check path
        # -ROOT
        assert os.path.exists(self.dataset_dir_path)
        assert os.path.exists(self.dataset_configuration_path)
        # -SRC
        assert os.path.exists(self.dataset_module_dir_path)
        assert os.path.exists(self.models_dir_path)
        assert os.path.exists(os.path.join(self.dataset_module_dir_path, self.DATASET_TYPE + ".py"))
        # -OUTPUTS
        if os.path.exists(self.OUTPUTS_DIR):
            r_v = input("La cartella di output esiste già. Sovrascriverla?"
                        "(Questo comporterà la perdita di tutti i dati Yes[Y] No[N]")
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
        self.G1_name_dir_test_inference = os.path.join(self.OUTPUTS_DIR, "inference_test_set", "G1")
        os.makedirs(self.G1_name_dir_test_inference, exist_ok=True)

    def load_evaluate_path_G1(self):
        self.G1_evaluation_path = os.path.join(self.OUTPUTS_DIR, "evaluation", "G1")
        os.makedirs(self.G1_evaluation_path, exist_ok=True)

    def load_train_path_GAN(self):
        self.GAN_logs_dir_path = os.path.join(self.OUTPUTS_DIR, "logs", "GAN")
        self.GAN_weights_path = os.path.join(self.OUTPUTS_DIR, "weights", "GAN")
        self.GAN_grid_path = os.path.join(self.OUTPUTS_DIR, "grid", "GAN")

        os.makedirs(self.GAN_logs_dir_path, exist_ok=True)
        os.makedirs(self.GAN_weights_path, exist_ok=True)
        os.makedirs(self.GAN_grid_path, exist_ok=True)

    def load_inference_path_GAN(self):
        self.GAN_name_dir_test_inference = os.path.join(self.OUTPUTS_DIR, "inference_test_set", "GAN")
        os.makedirs(self.GAN_name_dir_test_inference, exist_ok=True)

    def load_evaluate_path_GAN(self):
        self.GAN_evaluation_path = os.path.join(self.OUTPUTS_DIR, "evaluation", "GAN")
        os.makedirs(self.GAN_evaluation_path, exist_ok=True)

    def __load_dataset_info(self):
        # Se l assert va in errore il dataset non è presente
        assert os.path.exists(os.path.join(self.dataset_configuration_path, 'sets_config.pkl'))

        with open(os.path.join(self.dataset_configuration_path, 'sets_config.pkl'), 'rb') as f:
            dic = pickle.load(f)
            # nome sets
            self.name_tfrecord_train = os.path.join(self.dataset_configuration_path, dic['train']['name_file'])
            self.name_tfrecord_valid = os.path.join(self.dataset_configuration_path, dic['valid']['name_file'])
            self.name_tfrecord_test = os.path.join(self.dataset_configuration_path, dic['test']['name_file'])
            # numero di pair
            self.dataset_train_len = int(dic['train']['tot'])
            self.dataset_valid_len = int(dic['valid']['tot'])
            self.dataset_test_len = int(dic['test']['tot'])
            # lista pz presenti
            self.dataset_train_list = dic['train']['list_pz']  # pz presenti nel train
            self.dataset_valid_list = dic['valid']['list_pz']
            self.dataset_test_list = dic['test']['list_pz']