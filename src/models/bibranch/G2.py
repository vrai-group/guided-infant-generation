import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from src.models.Model_template import Model_Template

class G2(Model_Template):

    def __init__(self):
        self.architecture = "bi"
        self.input_shape = [96, 128, 2]
        self.output_channels = 1
        self.conv_hidden_num = 128
        self.repeat_num = int(np.log2(self.input_shape[0])) - 2
        self.activation_fn = 'relu'
        self.data_format = 'channels_last'
        self.lr_initial_G2 = 2e-5
        super().__init__()


    def _build_model(self):
        # Encoder
        inputs = Input(shape=self.input_shape)

        conv1 = Conv2D(self.conv_hidden_num, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(inputs)
        conv1 = Conv2D(self.conv_hidden_num, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv1)
        conv1 = Conv2D(self.conv_hidden_num, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv1)

        pool1 = Conv2D(self.conv_hidden_num * 2, 2, (2, 2), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv1)
        conv2 = Conv2D(self.conv_hidden_num * 2, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(pool1)
        conv2 = Conv2D(self.conv_hidden_num * 2, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv2)  # 256

        # Bridge
        pool3 = Conv2D(self.conv_hidden_num * 3, 2, (2, 2), activation=self.activation_fn,
                       data_format=self.data_format)(conv2)  # pool
        conv3 = Conv2D(self.conv_hidden_num * 3, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(pool3)
        conv3 = Conv2D(self.conv_hidden_num * 3, 3, (1, 1), padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv3)  # 384
        up4 = UpSampling2D(size=(2, 2), data_format=self.data_format, interpolation="nearest")(conv3)
        up4 = Conv2D(self.conv_hidden_num, 2, 1, padding="same", activation=self.activation_fn,
                     data_format=self.data_format)(up4)  # 128

        # Decoder
        merge4 = Concatenate(axis=-1)([up4, conv2])  # Long Skip connestion 128+256 =384
        conv4 = Conv2D(384, 3, 1, padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(merge4)
        conv4 = Conv2D(384, 3, 1, padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv4)

        up5 = UpSampling2D(size=(2, 2), data_format=self.data_format, interpolation="nearest")(conv4)
        up5 = Conv2D(self.conv_hidden_num, 2, 1, padding="same", activation=self.activation_fn,
                     data_format=self.data_format)(up5)
        merge5 = Concatenate(axis=-1)([up5, conv1])  # Long Skip connestion 128+128
        conv5 = Conv2D(256, 3, 1, padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(merge5)
        conv5 = Conv2D(256, 3, 1, padding='same', activation=self.activation_fn,
                       data_format=self.data_format)(conv5)

        outputs = Conv2D(self.output_channels, 1, 1, padding='same', activation=None,
                         data_format=self.data_format)(conv5)

        model = keras.Model(inputs, outputs)

        return model

    # Optimizer
    def _optimizer(self):
        return Adam(learning_rate=self.lr_initial_G2, beta_1=0.5)

    def prediction(self, I_PT1, Ic, Pt):
        # noise = (np.random.normal(0, 1, I_PT1.shape) * 0.0010) * tf.math.reduce_sum((Pt + 1) / 2, axis=-1).numpy().reshape(I_PT1.shape)
        # I_PT1 = tf.add(I_PT1, noise)
        input_G2 = tf.concat([I_PT1, Ic], axis=-1)  # [batch, 96, 128, 2]
        output_G2 = self.model(input_G2)  # [batch, 96, 128, 1] dtype=float32
        return output_G2

    # Loss
    def PoseMaskloss(self, I_PT2, It, Mt):
        It = tf.cast(It, dtype=tf.float32)
        I_PT2 = tf.cast(I_PT2, dtype=tf.float32)
        Mt = tf.cast(Mt, dtype=tf.float32)

        primo_membro = tf.reduce_mean(tf.abs(I_PT2 - It))  # L1 loss
        secondo_membro = tf.reduce_mean(tf.abs(I_PT2 - It) * Mt)
        loss = primo_membro + secondo_membro
        return loss

    def adv_loss(self, D_neg_refined_result, I_PT2, It, Mt):
        # Loss per imbrogliare il discriminatore creando un immagine sempre più reale
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_neg_refined_result,
                                                                          labels=tf.ones_like(
                                                                              D_neg_refined_result)))
        gen_cost = tf.cast(gen_cost, dtype=tf.float32)

        poseMaskLoss = self.PoseMaskloss2(I_PT2, It, Mt)

        loss = gen_cost + poseMaskLoss * 10

        return loss


    # Metriche
    def ssim(self, I_PT2, It, mean_0, mean_1, unprocess_function):
        It = tf.reshape(It, [-1, 96, 128, 1])
        I_PT2 = tf.reshape(I_PT2, [-1, 96, 128, 1])

        It = tf.cast(tf.clip_by_value(unprocess_function(It, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        I_PT2 = tf.cast(tf.clip_by_value(unprocess_function(I_PT2, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)

        result = tf.image.ssim(I_PT2, It, max_val=tf.reduce_max(It) - tf.reduce_min(It))
        mean = tf.reduce_mean(result)

        return mean


    def mask_ssim(self, I_PT2, It, Mt, mean_0, mean_1, unprocess_function):
        It = tf.reshape(It, [-1, 96, 128, 1])
        Mt = tf.reshape(Mt, [-1, 96, 128, 1])
        I_PT2 = tf.reshape(I_PT2, [-1, 96, 128, 1])

        It = tf.cast(tf.clip_by_value(unprocess_function(It, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        I_PT2 = tf.cast(tf.clip_by_value(unprocess_function(I_PT2, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        Mt = tf.cast(Mt, dtype=tf.uint16)

        mask_image_raw_1 = Mt * It
        mask_refined_result = Mt * I_PT2

        result = tf.image.ssim(mask_image_raw_1, mask_refined_result, max_val=tf.reduce_max(It) - tf.reduce_min(It))
        mean = tf.reduce_mean(result)

        return mean