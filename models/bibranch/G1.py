import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from models.Model_template import Model_Template

class G1(Model_Template):

    def __init__(self):
        self.input_shape = [96, 128, 15]
        self.output_channels = 1
        self.activation_fn = 'relu'
        self.lr_initial_G1 = 2e-5
        super().__init__() # eredita self.model e self.opt


    # MODEL
    def _build_model(self):
        inputs = Input(shape = self.input_shape)
        ##Encoder

        # Blocco 1
        Enc_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(inputs)
        Enc_1 = Activation(self.activation_fn)(Enc_1)

        branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
        branch_1_Enc_1 = Activation(self.activation_fn)(branch_1_Enc_1)

        branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_1_Enc_1)
        branch_1_Enc_1 = Activation(self.activation_fn)(branch_1_Enc_1)

        branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
        branch_2_Enc_1 = Activation(self.activation_fn)(branch_2_Enc_1)

        branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_2_Enc_1)
        branch_2_Enc_1 = Activation(self.activation_fn)(branch_2_Enc_1)

        concat_1 = concatenate([branch_1_Enc_1, branch_2_Enc_1])

        # Blocco 2
        Enc_2 = Conv2D(filters=256, kernel_size=2, strides=2)(concat_1)
        Enc_2 = Activation(self.activation_fn)(Enc_2)

        branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
        branch_1_Enc_2 = Activation(self.activation_fn)(branch_1_Enc_2)

        branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_1_Enc_2)
        branch_1_Enc_2 = Activation(self.activation_fn)(branch_1_Enc_2)

        branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
        branch_2_Enc_2 = Activation(self.activation_fn)(branch_2_Enc_2)

        branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_2_Enc_2)
        branch_2_Enc_2 = Activation(self.activation_fn)(branch_2_Enc_2)

        concat_2 = concatenate([branch_1_Enc_2, branch_2_Enc_2])

        # Blocco 3
        Enc_3 = Conv2D(filters=384, kernel_size=2, strides=2)(concat_2)
        Enc_3 = Activation(self.activation_fn)(Enc_3)

        branch_1_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(Enc_3)
        branch_1_Enc_3 = Activation(self.activation_fn)(branch_1_Enc_3)

        branch_1_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(branch_1_Enc_3)
        branch_1_Enc_3 = Activation(self.activation_fn)(branch_1_Enc_3)

        branch_2_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(Enc_3)
        branch_2_Enc_3 = Activation(self.activation_fn)(branch_2_Enc_3)

        branch_2_Enc_3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(branch_2_Enc_3)
        branch_2_Enc_3 = Activation(self.activation_fn)(branch_2_Enc_3)

        concat_3 = concatenate([branch_1_Enc_3, branch_2_Enc_3])

        # Blocco 4
        Enc_4 = Conv2D(filters=512, kernel_size=2, strides=2)(concat_3)
        Enc_4 = Activation(self.activation_fn)(Enc_4)

        branch_1_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(Enc_4)
        branch_1_Enc_4 = Activation(self.activation_fn)(branch_1_Enc_4)

        branch_1_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(branch_1_Enc_4)
        branch_1_Enc_4 = Activation(self.activation_fn)(branch_1_Enc_4)

        branch_2_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(Enc_4)
        branch_2_Enc_4 = Activation(self.activation_fn)(branch_2_Enc_4)

        branch_2_Enc_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(branch_2_Enc_4)
        branch_2_Enc_4 = Activation(self.activation_fn)(branch_2_Enc_4)

        concat_4 = concatenate([branch_1_Enc_4, branch_2_Enc_4]) #512

        ## Bridge
        # output [num_batch, 98.304] --> 98.304:  12x16x512
        x = Reshape([-1, int(np.prod([12, 16, 512]))])(concat_4)
        # output [num_batch, 64]
        z = Dense(64, activation=None)(x)
        # output [num batch, 24576]
        z = Dense(int(np.prod([12, 16, 128])), activation=None)(z)
        # output [num batch, 12,16,128]
        x = Reshape([12, 16, 128])(z)

        ##### Decoder
        # Blocco 1
        long_connection_1 = Concatenate(axis=-1)([x, concat_4]) #512+128=640

        branch_1_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(long_connection_1)
        branch_1_Dec_1 = Activation(self.activation_fn)(branch_1_Dec_1)

        branch_1_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(
            branch_1_Dec_1)
        branch_1_Dec_1 = Activation(self.activation_fn)(branch_1_Dec_1)

        branch_2_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(long_connection_1)
        branch_2_Dec_1 = Activation(self.activation_fn)(branch_2_Dec_1)

        branch_2_Dec_1 = Conv2D(filters=320, kernel_size=3, strides=1, padding='same')(
            branch_2_Dec_1)
        branch_2_Dec_1 = Activation(self.activation_fn)(branch_2_Dec_1)

        Dec_1 = concatenate([branch_1_Dec_1, branch_2_Dec_1]) #640
        Dec_1 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_1)
        Dec_1 = Conv2D(filters=384, kernel_size=1, strides=1, padding='same')(Dec_1)

        # Blocco 2
        long_connection_2 = Concatenate(axis=-1)([Dec_1, concat_3]) #384+384=768

        branch_1_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(long_connection_2)
        # branch_1_Dec_2 = BatchNormalization()(branch_1_Dec_2)
        branch_1_Dec_2 = Activation(self.activation_fn)(branch_1_Dec_2)

        branch_1_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(
            branch_1_Dec_2)
        branch_1_Dec_2 = Activation(self.activation_fn)(branch_1_Dec_2)

        branch_2_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(long_connection_2)
        branch_2_Dec_2 = Activation(self.activation_fn)(branch_2_Dec_2)

        branch_2_Dec_2 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(
            branch_2_Dec_2)
        branch_2_Dec_2 = Activation(self.activation_fn)(branch_2_Dec_2)

        Dec_2 = concatenate([branch_1_Dec_2, branch_2_Dec_2]) # 768
        Dec_2 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_2)
        Dec_2 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(Dec_2)

        # Blocco 3
        long_connection_3 = Concatenate(axis=-1)([Dec_2, concat_2])  # 512

        branch_1_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(long_connection_3)
        branch_1_Dec_3 = Activation(self.activation_fn)(branch_1_Dec_3)

        branch_1_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(
            branch_1_Dec_3)
        branch_1_Dec_3 = Activation(self.activation_fn)(branch_1_Dec_3)

        branch_2_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(long_connection_3)
        branch_2_Dec_3 = Activation(self.activation_fn)(branch_2_Dec_3)

        branch_2_Dec_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(
            branch_2_Dec_3)
        branch_2_Dec_3 = Activation(self.activation_fn)(branch_2_Dec_3)

        Dec_3 = concatenate([branch_1_Dec_3, branch_2_Dec_3])
        Dec_3 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_3)
        Dec_3 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(Dec_3)

        # Blocco 4
        long_connection_4 = Concatenate(axis=-1)([Dec_3, concat_1])

        branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
        branch_1_Dec_4 = Activation(self.activation_fn)(branch_1_Dec_4)

        branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
            branch_1_Dec_4)
        branch_1_Dec_4 = Activation(self.activation_fn)(branch_1_Dec_4)

        branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
        branch_2_Dec_4 = Activation(self.activation_fn)(branch_2_Dec_4)

        branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
            branch_2_Dec_4)
        branch_2_Dec_4 = Activation(self.activation_fn)(branch_2_Dec_4)

        Dec_4 = concatenate([branch_1_Dec_4, branch_2_Dec_4])  # 256

        outputs = Conv2D(1, 1, 1, padding='same', activation=None)(Dec_4)

        return Model(inputs, outputs)

    # Optimizer
    def _optimizer(self):
        return Adam(learning_rate=self.lr_initial_G1, beta_1=0.5, beta_2=0.999)

    def prediction(self, Ic, Pt):
        input_G1 = tf.concat([Ic, Pt], axis=-1)
        output_G1 = self.model(input_G1)  # [batch, 96, 128, 1] dtype=float32
        return output_G1

    # LOSS
    def PoseMaskloss1(self, output_G1, image_raw_1, mask_1):
        image_raw_1 = tf.cast(image_raw_1, dtype=tf.float32)
        mask_1 = tf.cast(mask_1, dtype=tf.float32)

        primo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1))  # L1 loss
        secondo_membro = tf.reduce_mean(tf.abs(output_G1 - image_raw_1) * mask_1)
        PoseMaskLoss1 = primo_membro + secondo_membro

        return PoseMaskLoss1

    # METRICHE
    def ssim(self, output_G1, image_raw_1, mean_0, mean_1, unprocess_function):
        image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
        output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
        output_G1 = tf.cast(output_G1, dtype=tf.float16)

        image_raw_1 = tf.cast(tf.clip_by_value(unprocess_function(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        output_G1 = tf.cast(tf.clip_by_value(unprocess_function(output_G1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)

        result = tf.image.ssim(output_G1, image_raw_1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
        mean = tf.reduce_mean(result)

        return mean

    def mask_ssim(self, output_G1, image_raw_1, mask_1, mean_0, mean_1, unprocess_function):
        image_raw_1 = tf.reshape(image_raw_1, [-1, 96, 128, 1])
        mask_1 = tf.reshape(mask_1, [-1, 96, 128, 1])
        output_G1 = tf.reshape(output_G1, [-1, 96, 128, 1])
        output_G1 = tf.cast(output_G1, dtype=tf.float16)

        image_raw_1 = tf.cast(tf.clip_by_value(unprocess_function(image_raw_1, mean_1, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        output_G1 = tf.cast(tf.clip_by_value(unprocess_function(output_G1, mean_0, 32765.5), clip_value_min=0, clip_value_max=32765), dtype=tf.uint16)
        mask_1 = tf.cast(mask_1, dtype=tf.uint16)

        mask_image_raw_1 = mask_1 * image_raw_1
        mask_output_G1 = mask_1 * output_G1
        result = tf.image.ssim(mask_image_raw_1, mask_output_G1, max_val=tf.reduce_max(image_raw_1) - tf.reduce_min(image_raw_1))
        mean = tf.reduce_mean(result)
        mean = tf.cast(mean, dtype=tf.float32)

        return mean




