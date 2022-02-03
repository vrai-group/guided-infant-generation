import os
import math
import numpy as np
import importlib.util
from PIL import Image
import tensorflow as tf


####################################
# Utils per la creazione dei TFRecords
####################################

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
  """Returns a TF-Feature of float32.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


####################################
# Utils pda utilizzare nel training
####################################
"""
Questo metodo consente di crere una griglia
"""
def save_grid(tensor, filename, nrow=8, padding=2, normalize=False, scale_each=False):
    def _grid(tensor, nrow=8, padding=2, normalize=False, scale_each=False):
        """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
        nmaps = tensor.shape[0]
        xmaps = min(nrow, nmaps)  # numero di colonne
        ymaps = int(math.ceil(float(nmaps) / xmaps))  # numero di righe
        height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
        grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2], dtype=np.uint8)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                h, h_width = y * height + 1 + padding // 2, height - padding
                w, w_width = x * width + 1 + padding // 2, width - padding
                grid[h:h + h_width, w:w + w_width] = tf.reshape(tensor[k], (96, 128))
                k = k + 1
        return grid

    ndarr = _grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)



"""
Questo metodo mi consente di caricare in maniera dinamica i vari moduli di riferimento per G1, G2, D, Syntetich.
Ad esempio: models/mono/G1.py
Ad esempio: dataset/Syntetich.py
"""
def import_module(name_module, path):
    spec = importlib.util.spec_from_file_location(name_module, os.path.join(path, name_module + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
