import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from models.Model_template import Model_Template

class G1(Model_Template):

    def __init__(self):
        self.architecture = "mono"
        self.input_shape = [96, 128, 15]
        self.output_channels = 1
        self.conv_hidden_num = 128
        self.repeat_num = int(np.log2(self.input_shape[0])) - 2
        self.activation_fn = 'relu'
        self.data_format = 'channels_last'
        self.lr_initial_G1 = 2e-5
        super().__init__() # eredita self.model e self.opt


    # MODEL
    def _build_model(self):

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

    # Optimizer
    def _optimizer(self):
        return Adam(learning_rate=self.lr_initial_G1, beta_1=0.5, beta_2=0.999)

    def prediction(self, Ic, Pt):
        input_G1 = tf.concat([Ic, Pt], axis=-1)
        output_G1 = self.model(input_G1)  # [batch, 96, 128, 1] dtype=float32
        output_G1 = tf.cast(output_G1, dtype=tf.float16)
        return output_G1

    # LOSS
    def PoseMaskloss(self, I_PT1, It, Mt):
        I_PT1 = tf.cast(I_PT1, dtype=tf.float32)
        It = tf.cast(It, dtype=tf.float32)
        Mt = tf.cast(Mt, dtype=tf.float32)

        primo_membro = tf.reduce_mean(tf.abs(I_PT1 - It))  # L1 loss
        secondo_membro = tf.reduce_mean(tf.abs(I_PT1 - It) * Mt)
        loss = primo_membro + secondo_membro

        return loss

    # METRICHE
    def ssim(self, I_PT1, It, mean_0, mean_1, unprocess_function):
        It = tf.reshape(It, [-1, 96, 128, 1])
        I_PT1 = tf.reshape(I_PT1, [-1, 96, 128, 1])
        I_PT1 = tf.cast(I_PT1, dtype=tf.float16)

        It = tf.cast(tf.clip_by_value(unprocess_function(It, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        I_PT1 = tf.cast(tf.clip_by_value(unprocess_function(I_PT1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)

        result = tf.image.ssim(I_PT1, It, max_val=tf.reduce_max(It) - tf.reduce_min(It))
        mean = tf.reduce_mean(result)

        return mean

    def mask_ssim(self, I_PT1, It, Mt, mean_0, mean_1, unprocess_function):
        It = tf.reshape(It, [-1, 96, 128, 1])
        Mt = tf.reshape(Mt, [-1, 96, 128, 1])
        I_PT1 = tf.reshape(I_PT1, [-1, 96, 128, 1])
        I_PT1 = tf.cast(I_PT1, dtype=tf.float16)

        It = tf.cast(tf.clip_by_value(unprocess_function(It, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        I_PT1 = tf.cast(tf.clip_by_value(unprocess_function(I_PT1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        Mt = tf.cast(Mt, dtype=tf.uint16)

        mask_image_raw_1 = Mt * It
        mask_output_G1 = Mt * I_PT1

        result = tf.image.ssim(mask_image_raw_1, mask_output_G1, max_val=tf.reduce_max(It) - tf.reduce_min(It))
        mean = tf.reduce_mean(result)
        mean = tf.cast(mean, dtype=tf.float32)

        return mean










