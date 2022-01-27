import os
from pg2 import PG2
from CONFIG import Config


if __name__ == "__main__":
    config = Config()
    config.save_config()

    pg2 = PG2(config)  # Pose Guided ^2 network

    if config.MODE == "train_G1":
        pg2.train_G1()
    elif config.MODE == "train_cDCGAN":
        pg2.train_cDCGAN()
    elif config.MODE == 'inference_G1':
        pg2.inference_on_test_set_G1()
    elif config.MODE == 'inference_G2':
        pg2.inference_on_test_set_G2()
    elif config.MODE == 'evaluate':
        pg2.evaluate_G1()
        pg2.evaluate_GAN()
        pg2.tsne()

