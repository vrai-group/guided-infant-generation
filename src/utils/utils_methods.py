import os
import math
import numpy as np
import importlib.util
from PIL import Image
import tensorflow as tf

##################################################
#
###################################################
"""
# Dato un radius di 4 e un punto p ciò che cerco di fare è di trovare tutti i punti che si trovano 
# nell'intorno [-4,4] del punto p. Le coordinate di ognuno di questi punti le salvo in indices e setto il valore 1 (visibile).
# Al termine, ciò che otteniamo è che il punto p viene ingrandito considerando un raggio di 4.
# Un esempio è mostrato in figura SparseKeypoint.png
"""
def getSparseKeypoint(y, x, k, height, width, radius=4, mode='Solid'):
    indices = []
    values = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            distance = np.sqrt(float(i ** 2 + j ** 2))
            if y + i >= 0 and y + i < height and x + j >= 0 and x + j < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([y + i, x + j, k])
                    values.append(1)
                    # dense = np.squeeze(_sparse2dense(indices, values, [height, width, 1]))
                    # cv2.imwrite('SparseKeypoint.png', dense * 255)

    return indices, values


"""
# Andiamo ad ingrandire ogni peaks di un raggio 4 o superiore creando nuovi punti.
# Salviamo tutti in indices. I Values sono settati ad 1 ed indicano la visibilità degli indices
# i valori di k indicano gli indici di ogni keypoint:
  0 head; 1 right_hand; 2 right_elbow; 3 right_shoulder; 4 neck; 5 left_shoulder; 6 left_elbow;
  7 left_hand; 8 right_foot; 9 right_knee; 10 right_hip; 11 left_hip; 12 left_knee; 13 left_foot

@:return
indices --> [ [<coordinata>, <coordinata>, <indice keypoint>], ... ]
values --> [  1,1,1, ... ]
shape -->  lista [height, width, num keypoints]
"""
def getSparsePose(peaks, height, width, radius, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]  # coordinate peak ex: "300,200"
        x = p[0]
        y = p[1]
        if x != -1 and y != -1:  # non considero le occlusioni indicate con -1
            ind, val = _getSparseKeypoint(y, x, k, height, width, radius, mode)
            indices.extend(ind)
            values.extend(val)
    return indices, values

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

"""
Crezione dell example da aggiungere al TF Record
@:return example
"""
def format_example(dic):
    example = tf.train.Example(features=tf.train.Features(feature={


        'pz_condition': bytes_feature(dic["pz_condition"].encode('utf-8')),  # nome del pz di condizione
        'pz_target': bytes_feature(dic["pz_target"].encode('utf-8')), # nome del pz di target

        'Ic_image_name': bytes_feature(dic["Ic_image_name"].encode('utf-8')),  # nome dell immagine di condizione
        'It_image_name': bytes_feature(dic["It_image_name"].encode('utf-8')),  # nome dell immagine di target
        'Ic': bytes_feature(dic["Ic"].tostring()),  # immagine di condizione
        'It': bytes_feature(dic["It"].tostring()),  # immagine target

        'image_format': bytes_feature(dic['image_format'].encode('utf-8')),
        'image_height': int64_feature(dic['image_height']),
        'image_width': int64_feature(dic['image_width']),

        # valori delle coordinate originali della posa ridimensionati a 96x128
        'Ic_original_keypoints': bytes_feature(np.array(dic["Ic_original_keypoints"]).astype(np.int64).tostring()),
        'It_original_keypoints': bytes_feature(np.array(dic["It_original_keypoints"]).astype(np.int64).tostring()),
        'shape_len_Ic_original_keypoints': int64_feature(np.array(dic["Ic_original_keypoints"]).shape[0]),
        'shape_len_It_original_keypoints': int64_feature(np.array(dic["It_original_keypoints"]).shape[0]),

        # maschera binaria a radius (r_k) con shape [96, 128, 1]
        'Mc': int64_feature(dic["Mc"].astype(np.uint8).flatten().tolist()),
        'Mt': int64_feature(dic["Mt"].astype(np.uint8).flatten().tolist()),

        # Sparse tensor per la posa. Gli indici e i valori considerano il riempimento (ingrandimento) del Keypoints di raggio r_k
        'Ic_indices': bytes_feature(np.array(dic["Ic_indices"]).astype(np.int64).tostring()),
        'Ic_values': bytes_feature(np.array(dic["Ic_values"]).astype(np.int64).tostring()),
        'It_indices': bytes_feature(np.array(dic["It_indices"]).astype(np.int64).tostring()),
        'It_values': bytes_feature(np.array(dic["It_values"]).astype(np.int64).tostring()),
        'shape_len_Ic_indices': int64_feature(np.array(dic["Ic_indices"]).shape[0]),
        'shape_len_It_indices': int64_feature(np.array(dic["It_indices"]).shape[0]),

        'radius_keypoints': int64_feature(dic['radius_keypoints']),


    }))

    return example


####################################
# Utils da utilizzare nel training
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
