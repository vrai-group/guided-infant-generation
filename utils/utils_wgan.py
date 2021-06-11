__author__ = 'shekkizh'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys

def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm


def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel



