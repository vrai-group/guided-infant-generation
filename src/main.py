import os
from pg2 import PG2
from CONFIG import Config


if __name__ == "__main__":
    config = Config()

    pg2 = PG2(config)  # Pose Guided ^2 network

    if config.MODE == "train_G1":
        pg2.train_G1()
    elif config.MODE == "train_cDCGAN":
        pg2.train_cDCGAN()
    elif config.MODE == "plot_history_G1":
        pg2.plot_history_G1()
    elif config.MODE == "plot_history_GAN":
        pg2.plot_history_GAN()
    elif config.MODE == 'evaluate_G1':
        pg2.evaluate_G1(analysis_set="test_set", name_dataset=config.name_tfrecord_test, dataset_len=config.dataset_test_len)
    elif config.MODE == 'evaluate_GAN':
        pg2.evaluate_GAN(analysis_set="test_set", name_dataset=config.name_tfrecord_test, dataset_len=config.dataset_test_len)
    elif config.MODE == 'tsne_GAN':
        pg2.tsne(key_image_interested="test_20")  # id immagine ottenuto dalla inference
    elif config.MODE == 'inference_G1':
        pg2.inference_on_test_set_G1()
    elif config.MODE == 'inference_GAN':
        pg2.inference_on_test_set_G2()