import os
from pg2 import PG2
from CONFIG import Config


if __name__ == "__main__":
    config = Config()
    config.save_config()

    G1_NAME_WEIGHTS_FILE = '..\\weights\\Model_G1_Bibranch_epoch_005-loss_0.000-ssim_0.943-mask_ssim_0.984-val_loss_0.001-val_ssim_0.917-val_mask_ssim_0.979.hdf5'
    G2_NAME_WEIGHTS_FILE = '..\\weights\\Model_G2_Bibranch_epoch_184-loss_0.69-ssmi_0.93-mask_ssmi_1.00-r_r_5499-im_0_5484-im_1_5464-val_loss_0.70-val_ssim_0.77-val_mask_ssim_0.98-val_r_r_400-val_im_0_400-val_im_1_400.hdf5'

    pg2 = PG2(config)  # Pose Guided ^2 network

    if config.MODE == "train_G1":
        pg2.train_G1()
    elif config.MODE == "train_cDCGAN":
        pg2.train_cDCGAN()
    elif config.MODE == 'evaluate':
        pg2.evaluate_G1(analysis_set="test_set", name_dataset=config.name_tfrecord_test, dataset_len=config.dataset_test_len)
        pg2.evaluate_GAN(analysis_set="test_set", name_dataset=config.name_tfrecord_test, dataset_len=config.dataset_test_len)
        pg2.tsne(key_image_interested="test_20")  # id immagine ottenuto dalla inference
    elif config.MODE == 'inference_G1':
        pg2.inference_on_test_set_G1(G1_NAME_WEIGHTS_FILE, save_figure=False)
    elif config.MODE == 'inference_G2':
        pg2.inference_on_test_set_G2(G1_NAME_WEIGHTS_FILE, G2_NAME_WEIGHTS_FILE, save_figure=False)

