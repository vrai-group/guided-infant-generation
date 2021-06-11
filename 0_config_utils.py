import os
import numpy as np

class Config:
    def __init__(self) :

        #Path
        self.data_path = './data/BabyPose'  # dove si trova il dataset
        self.data_annotations_path =  './data/annotations'
        self.weigths_path = './weights'
        self.logs_path = './logs'

        #General
        self.img_H = 128 #'input image height' 
        self.img_W = 64 #'input image width'
        self.dataset_train_len = 25600  # numero di pair nel train
        self.dataset_valid_len = 12800  # numero di pair nel valid
        self.dataset_test_len = 12800   # numero di pair nel test

        #model G1 / G2
        self.input_shape_g1 = [128, 64, 21]  # concat tra image_raw_0 a 3 channel e la posa a 18 channel
        self.input_shape_g2 = [128, 64, 6]  # concat tra image_raw_0 a 3 channel e l' output del generatore G1
        self.repeat_num = int(np.log2(self.img_H)) - 2  # numero di blocchi residuali --> 5 con height 128
        self.conv_hidden_num = 128 # numero di filtri del primo layer convoluzionale. dall helper n in the paper
        self.z_num = 64 # numero neuroni del fully connected per ora solo del G1
        self.input_image_raw_channel = 3  # indica per le image_raw_0 il 3 canali RGB, mi serve per la regressione di output della rete
        self.activation_fn = 'relu'
        self.min_fea_map_H = 8
        self.keypoint_num = 14  # numero di mappe
        self.dataset = 'Market_train'  # data\Market1501_img_pose_attr_seg\Market_train_data
        self.split= 'train'

        # Training / test parameters
        self.is_train = False
        self.epochs = 100
        self.batch_size_train = 1  # grandezza del batch_size
        self.batch_size_valid = 16  # grandezza del batch_size
        self.lr_update_epoch = 10  # epoche di aggiornameto del learning rate
        self.d_lr = 0.00008 # learning_rate del discriminatore
        self.g_lr=0.00008   # learning rate del generatore G2
        self.run_google_colab = True
        self.download_weight = 5 # step di epoche in cui verrà effettuato il download

        self.beta1=0.5   # adam parameter
        self.beta2=0.999   # adam parametr
        self.gamma=0.5 
        self.lambda_k=0.001 
        self.use_gpu=False
        self.gpu=-1
        if self.use_gpu:
            self.data_format = 'channels_first'
        else:
            self.data_format = 'channels_last' #'NHWC'

        if self.is_train:
            self.num_threads = 4
            self.capacityCoff = 2  # serve per indicare quanti batch devono esssere caricati in memoria
        else:  # during testing to keep the order of the input data
            self.num_threads = 1
            self.capacityCoff = 1

        self.model=0
        self.D_arch = 'DCGAN'   # 'DCGAN'  'noNormDCGAN'  'MultiplicativeDCGAN'  'tanhNonlinearDCGAN'  'resnet101'

        self.load_path=''
        self.log_step=200 
        self.save_model_secs=1000 
        self.num_log_samples=3 
        self.log_level='INFO'
        self.log_dir='logs' 
        self.model_dir=None
        self.test_data_path=None #help='directory with images which will be used in test sample generation'
        self.sample_per_image =64 #help='# of sample per image during test sample generation'
        self.random_seed=123
