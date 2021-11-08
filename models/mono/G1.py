import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from models.Model_template import Model_Template

class G1(Model_Template):

    def __init__(self):
        #TODO: RIchiamare il costruttore super
        self.input_shape = [96, 128, 15]
        self.output_channels = 1
        self.conv_hidden_num = 128
        self.repeat_num = int(np.log2(self.input_shape[0])) - 2
        self.activation_fn = 'relu'
        self.data_format = 'channels_last'
        self.lr_initial_G1 = 2e-5

    # MODEL
    def build_model(self):

        # Encoder
        encoder_layer_list = []
        inputs = Input(shape=self.input_shape)
        x = Conv2D(self.conv_hidden_num, 3, (1, 1), padding='same', activation=self.activation_fn, data_format=self.data_format)(inputs)

        for idx in range(self.repeat_num):
            channel_num = self.conv_hidden_num * (idx + 1)
            res = x
            x = Conv2D(channel_num, 3, (1, 1), padding='same', activation=self.activation_fn, data_format=self.data_format)(x)
            x = Conv2D(channel_num, 3, (1, 1), padding='same', activation=self.activation_fn, data_format=self.data_format)(x)
            x = Add()([x, res])
            encoder_layer_list.append(x)
            if idx < self.repeat_num - 1:
                x = Conv2D(self.conv_hidden_num * (idx + 2), 2, (2, 2), activation=self.activation_fn, data_format=self.data_format)(x)

        # Bridge
        x = Reshape([-1, int(np.prod([12, 16, channel_num]))])(x)
        z = Dense(64, activation=None)(x)

        x = Dense(int(np.prod([12, 16, self.conv_hidden_num])), activation=None)(z)
        x = Reshape([12, 16, self.conv_hidden_num])(x)

        # Decoder
        for idx in range(self.repeat_num):

            x = Concatenate(axis=-1)([x, encoder_layer_list[self.repeat_num - 1 - idx]])  # Long Skip connestion
            res = x
            channel_num = x.get_shape()[-1]
            x = Conv2D(channel_num, 3, 1, padding='same', activation=self.activation_fn, data_format=self.data_format)(x)
            x = Conv2D(channel_num, 3, 1, padding='same', activation=self.activation_fn, data_format=self.data_format)(x)
            x = Add()([x, res])
            if idx < self.repeat_num - 1:
                x = UpSampling2D(size=(2, 2), data_format=self.data_format, interpolation="nearest")(x)
                x = Conv2D(self.conv_hidden_num  * (self.repeat_num - idx - 1), 1, 1, activation=self.activation_fn, data_format=self.data_format)(x)

        outputs = Conv2D(self.output_channels, 3, 1, padding='same', activation=None, data_format=self.data_format)(x)

        return Model(inputs, outputs)

    # LOSS
    def PoseMaskloss(self, output_G1, image_raw_1, mask_1):
        image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)
        mask_1 = tf.cast(mask_1, dtype=tf.float32)

        primo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1))  # L1 loss
        secondo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1) * mask_1)
        PoseMaskLoss1 = primo_membro + secondo_membro

        return PoseMaskLoss1

    # Learning rate
    def step_decay(self):
        pass

    # Optimizer
    def optimizer(self):
        return Adam(learning_rate=self.lr_initial_G1, beta_1=0.5, beta_2=0.999)

    # METRICHE
    def m_ssim(self, output_G1, image_raw_1, mean_0, mean_1):
        image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
        output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
        output_G1 = tf.cast(output_G1, dtype=tf.float16)

        image_raw_1 = tf.cast(tf.clip_by_value(self.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        output_G1 = tf.cast(tf.clip_by_value(self.unprocess_image(output_G1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)

        result = tf.image.ssim(output_G1, image_raw_1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
        mean = tf.reduce_mean(result)

        return mean

    def mask_ssim(self, output_G1, image_raw_1, mask_1, mean_0, mean_1):
        image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
        mask_1 = tf.reshape(mask_1, [-1, 96, 128, 1])
        output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
        output_G1 = tf.cast(output_G1, dtype=tf.float16)

        image_raw_1 = tf.cast(tf.clip_by_value(self.unprocess_image(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        output_G1 = tf.cast(tf.clip_by_value(self.unprocess_image(output_G1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        mask_1 = tf.cast(mask_1, dtype=tf.uint16)

        mask_image_raw_1 = mask_1 * image_raw_1
        mask_output_G1 = mask_1 * output_G1

        result = tf.image.ssim(mask_image_raw_1, mask_output_G1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
        mean = tf.reduce_mean(result)
        mean = tf.cast(mean, dtype=tf.float32)

        return mean










