import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt

from utils import grid
from utils import utils_wgan
from utils import dataset_utils
from model import G1, G2, Discriminator
from datasets.BabyPose import BabyPose
from Augumentation import apply_augumentation

Config_file = __import__('1_config_utils')
config = Config_file.Config()
babypose_obj = BabyPose()

dataset = babypose_obj.get_unprocess_dataset(config.name_tfrecord_valid)
dataset = dataset.batch(1)
it = iter(dataset)
name_tfrecord_aug, dataset_train_aug_len = apply_augumentation(it, config, "valid")

print(dataset_train_aug_len)