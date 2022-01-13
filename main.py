from pg2 import PG2
from CONFIG import Config

if __name__ == "__main__":
    config = Config()
    config.save_config()

    pg2 = PG2(config)  # Pose Guided ^2 network

    #pg2.train_G1()
    #pg2.train_cDCGAN()
    #pg2.prediction()