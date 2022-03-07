from pg2 import PG2
from config import Config

if __name__ == "__main__":

    config = Config()
    pg2 = PG2(config)  # Pose Guided network

    if config.MODE == "train_G1":
        pg2.train_G1()
    elif config.MODE == "train_cDCGAN":
        pg2.train_cDCGAN()
    elif config.MODE == "plot_history_G1":
        pg2.plot_history_G1()
    elif config.MODE == "plot_history_GAN":
        pg2.plot_history_GAN()
    elif config.MODE == 'evaluate_G1':
        pg2.evaluate_G1(analysis_set="test_set", name_dataset=config.name_tfrecord_test,
                        dataset_len=config.dataset_test_len)
    elif config.MODE == 'evaluate_GAN':
        pg2.evaluate_GAN(analysis_set="test_set", name_dataset=config.name_tfrecord_test,
                         dataset_len=config.dataset_test_len)
    elif config.MODE == 'tsne_GAN':
        # The dic_history_key_pair refers to a specific pair.
        # In this case, 'test_20', signify that we want compare the real and generated features of
        # 20th pair in test set. You can write other pairs. In particular, the value that you can use are defined in
        # dic_history.pkl file of your dataset configuration as keys of dictionary
        pg2.tsne(dic_history_key_pair="test_20")
    elif config.MODE == 'inference_G1':
        pg2.inference_on_test_set_G1()
    elif config.MODE == 'inference_GAN':
        pg2.inference_on_test_set_G2()
