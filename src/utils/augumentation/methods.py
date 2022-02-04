import cv2
import math
import numpy as np
import tensorflow as tf

from src.utils.utils_methods import getSparsePose

# STRUCTURAL
# luminoità tra max//3 e -max//3
def random_brightness(dic_data, indx_img):
    image = dic_data["I" + indx_img]
    max = np.int64(image.max())
    max_d = max // 3
    min_d = -max_d
    brightness = tf.random.uniform(shape=[1], minval=min_d, maxval=max_d, dtype=tf.dtypes.int64).numpy()

    new_image = image + brightness
    new_image = np.clip(new_image, 0, 32765)
    new_image = new_image.astype(np.uint16)
    dic_data["I" + indx_img] = new_image

    return dic_data

# contrasto tra max// e -max//2
def random_contrast(dic_data, indx_img):
    image = dic_data["I" + indx_img]
    max = np.int64(image.max())
    max_d = max // 2
    min_d = -max_d
    contrast = tf.random.uniform(shape=[1], minval=min_d, maxval=max_d, dtype=tf.dtypes.int64).numpy()

    f = (max + 4) * (contrast + max) / (max * ((max + 4) - contrast))
    alpha = f
    gamma = (max_d) * (1 - f)

    new_image = (alpha * image) + gamma
    new_image = np.clip(new_image, 0, 32765)
    new_image = new_image.astype(np.uint16)
    dic_data["I" + indx_img] = new_image

    return dic_data


# AFFINE
def aug_shift(dic_data, type, indx_img, tx=0, ty=0):
    if type == "or":
        assert (ty == 0)
    elif type == "ver":
        assert (tx == 0)

    h, w, c = dic_data["I" + indx_img].shape

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dic_data["I" + indx_img] = cv2.warpAffine(dic_data["I" + indx_img], M, (w, h),
                                                            flags=cv2.INTER_NEAREST,
                                                            borderMode=cv2.BORDER_REPLICATE).reshape(h, w, c)

    dic_data["M" + indx_img] = cv2.warpAffine(dic_data["M" + indx_img], M, (w, h),
                                                               flags=cv2.INTER_NEAREST,
                                                               borderMode=cv2.BORDER_REPLICATE).reshape(h, w, c)

    keypoints_shifted = []
    values_shifted = []
    for coordinates in dic_data["I"+indx_img+"indices"]:
        y, x, id = coordinates

        if type == "or":
            xs = x + tx
            ys = y
            if xs > 0 and xs < w:
                keypoints_shifted.append([ys, xs, id])
                values_shifted.append(1)

        elif type == "ver":
            xs = x
            ys = y + ty
            if ys > 0 and ys < h:
                keypoints_shifted.append([ys, xs, id])
                values_shifted.append(1)

    dic_data["I"+indx_img+"indices"] = keypoints_shifted
    dic_data["I"+indx_img+"values"] = values_shifted

    return dic_data

def aug_rotation_angle(dic_data, angle_deegre, indx_img):
    h, w, c = dic_data["I"+indx_img].shape
    ym, xm = h // 2, w // 2  # midpoint dell'immagine 96x128
    angle_radias = math.radians(angle_deegre)  # angolo di rotazione

    def rotate_keypoints(indices):
        keypoints_rotated = []
        for coordinates in indices:
            x, y = coordinates
            if y != -1 and x != -1:
                xr = (x - xm) * math.cos(angle_radias) - (y - ym) * math.sin(angle_radias) + xm
                yr = (x - xm) * math.sin(angle_radias) + (y - ym) * math.cos(angle_radias) + ym
                keypoints_rotated.append([int(xr), int(yr)])
            else:
                keypoints_rotated.append([x, y])

        return keypoints_rotated

    # If the angle is positive, the image gets rotated in the counter-clockwise direction.
    M = cv2.getRotationMatrix2D((xm, ym), -angle_deegre, 1.0)
    # Rotate image
    dic_data["I"+indx_img] = cv2.warpAffine(dic_data["I"+indx_img], M, (w, h), flags=cv2.INTER_NEAREST,
                                                            borderMode=cv2.BORDER_REPLICATE).reshape(h, w, c)

    # Rotate mask
    dic_data["M"+indx_img] = cv2.warpAffine(dic_data["M"+indx_img], M, (w, h)).reshape(h, w, c)

    # Rotate keypoints coordinate
    keypoints_rotated = rotate_keypoints(dic_data["I"+indx_img+"_original_keypoints"])
    dic_data["I"+indx_img+"_indices"], dic_data["I"+indx_img+"_values"] = getSparsePose(keypoints_rotated, h, w,
                                                                                         radius=dic_data['radius_keypoints'],
                                                                                         mode='Solid')

    return dic_data

def aug_flip(dic_data):
    ### Flip vertical pz_0
    mapping = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 10: 11, 11: 10, 9: 12, 12: 9, 8: 13, 13: 8}
    dic_data["Ic"] = cv2.flip(dic_data["Ic"], 1)
    dic_data["Ic_indices"] = [[i[0], 64 + (64 - i[1]), mapping[i[2]]] for i in dic_data["Ic_indices"]]
    dic_data["Mc"] = cv2.flip(dic_data["Mc"], 1)

    ### Flip vertical pz_1
    mapping = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 10: 11, 11: 10, 9: 12, 12: 9, 8: 13, 13: 8}
    dic_data["It"] = cv2.flip(dic_data["It"], 1)
    dic_data["It_indices"] = [[i[0], 64 + (64 - i[1]), mapping[i[2]]] for i in dic_data["It_indices"]]
    dic_data["Mt"] = cv2.flip(dic_data["Mt"], 1)

    return dic_data