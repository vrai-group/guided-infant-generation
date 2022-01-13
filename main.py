import os
from pg2 import PG2
from CONFIG import Config
from utils.utils_methods import import_module

if __name__ == "__main__":
    config = Config()
    config.save_config()

    # -Import dinamico dell modulo di preprocess dataset
    # Ad esempio: Syntetich
    name_module_preprocess_dataset = config.DATASET.split('_')[0]
    dataset_module = import_module(name_module_preprocess_dataset, config.dataset_module_dir_path)

    # -Import dinamico dell'architettura
    G1 = import_module(name_module="G1", path=config.models_dir_path).G1()
    G2 = import_module(name_module="G2", path=config.models_dir_path).G2()
    D = import_module(name_module="D", path=config.models_dir_path).D()

    config.G1_NAME_WEIGHTS_FILE = 'Model_G1_epoch_008-loss_0.000301-ssim_0.929784-mask_ssim_0.979453-val_loss_0.000808-val_ssim_0.911077-val_mask_ssim_0.972699.hdf5'
    config.G2_NAME_WEIGHTS_FILE = 'Model_G2_epoch_162-loss_0.69-ssmi_0.93-mask_ssmi_1.00-r_r_5949-im_0_5940-im_1_5948-val_loss_0.70-val_ssim_0.77-val_mask_ssim_0.98.hdf5'

    config.G1_weigths_file_path = os.path.join(config.G1_weigths_dir_path, config.G1_NAME_WEIGHTS_FILE)  # nome file pesi G1
    config.G2_weigths_file_path = os.path.join(config.GAN_weigths_dir_path, config.G2_NAME_WEIGHTS_FILE)  # nome file pesi G2

    
    pg2 = PG2(config, dataset_module=dataset_module, G1=G1, G2=G2, D=D)  # Pose Guided ^2 network

    #pg2.train_G1()
    #pg2.train_cDCGAN()
    #pg2.prediction()