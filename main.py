from pg2 import PG2
from CONFIG import Config

if __name__ == "__main__":
    config = Config()
    config.print_info()
    config.save_info()

    pg2 = PG2(config)  # Pose Guided ^2 network

    #pg2.train_G1()
    #pg2.train_cDCGAN()