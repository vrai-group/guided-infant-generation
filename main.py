from pg2 import PG2
from CONFIG import Config

if __name__ == "__main__":
    config = Config()
    config.save_config()

    # TODO: inserire il join del path
    name_weights_file_G1 = 'Model_G1_epoch_008-loss_0.000301-ssim_0.929784-mask_ssim_0.979453-val_loss_0.000808-val_ssim_0.911077-val_mask_ssim_0.972699.hdf5'
    name_weights_file_G2 = 'Model_G2_epoch_162-loss_0.69-ssmi_0.93-mask_ssmi_1.00-r_r_5949-im_0_5940-im_1_5948-val_loss_0.70-val_ssim_0.77-val_mask_ssim_0.98.hdf5'

    pg2 = PG2(config, name_weights_file_G1, name_weights_file_G2, "")  # Pose Guided ^2 network

    #pg2.train_G1()
    #pg2.train_cDCGAN()