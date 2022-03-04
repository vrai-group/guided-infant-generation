import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from models.Model_template import Model_Template

class G2(Model_Template):

    def __init__(self):
        self.architecture = "bibranch"
        self.input_shape = [96, 128, 2]
        self.output_channels = 1
        self.activation_fn = 'relu'
        self.lr_initial_G2 = 2e-5
        super().__init__() # eredita self.model e self.opt

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        ## Encoder

        Enc_1 = Conv2D(128, 3, (1, 1), padding='same', activation=self.activation_fn)(inputs)

        branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
        branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)

        branch_1_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_1_Enc_1)
        branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)

        branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(Enc_1)
        branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)

        branch_2_Enc_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(branch_2_Enc_1)
        branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)

        concat_1 = concatenate([branch_1_Enc_1, branch_2_Enc_1])  # 128

        Enc_2 = Conv2D(filters=256, kernel_size=2, strides=2)(concat_1)
        Enc_2 = Activation('relu')(Enc_2)

        branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
        branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)

        branch_1_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_1_Enc_2)
        branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)

        branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(Enc_2)
        branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)

        branch_2_Enc_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(branch_2_Enc_2)
        branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)

        concat_2 = concatenate([branch_1_Enc_2, branch_2_Enc_2])

        ## Bridge
        pool3 = Conv2D(384, 2, (2, 2), activation=self.activation_fn)(concat_2)  # pool
        conv3 = Conv2D(384, 3, (1, 1), padding='same', activation=self.activation_fn)(pool3)
        conv3 = Conv2D(384, 3, (1, 1), padding='same', activation=self.activation_fn)(conv3)  # 384
        up4 = UpSampling2D(size=(2, 2), interpolation="nearest")(conv3)
        up4 = Conv2D(128, 2, 1, padding="same", activation=self.activation_fn)(up4)  # 128

        ## Decoder

        # Blocco1
        long_connection_1 = Concatenate(axis=-1)([up4, concat_2])  # 128+256=384

        branch_1_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(long_connection_1)
        branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)

        branch_1_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(
            branch_1_Dec_1)
        branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)

        branch_2_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(long_connection_1)
        branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)

        branch_2_Dec_1 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same')(
            branch_2_Dec_1)
        branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)

        Dec_1 = concatenate([branch_1_Dec_1, branch_2_Dec_1])
        Dec_1 = UpSampling2D(size=(2, 2), interpolation="nearest")(Dec_1)
        Dec_1 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(Dec_1)

        # Blocco2
        long_connection_4 = Concatenate(axis=-1)([Dec_1, concat_1])  # 128+128=256

        branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
        branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)

        branch_1_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
            branch_1_Dec_4)
        branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)

        branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(long_connection_4)
        branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)

        branch_2_Dec_4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(
            branch_2_Dec_4)
        branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)

        Dec_4 = concatenate([branch_1_Dec_4, branch_2_Dec_4])  # 256

        outputs = Conv2D(1, 1, 1, padding='same', activation=None)(Dec_4)

        model = keras.Model(inputs, outputs)

        return model

    def _optimizer(self):
        return Adam(learning_rate=self.lr_initial_G2, beta_1=0.5)

    def prediction(self, I_PT1, Ic):
        input_G2 = tf.concat([I_PT1, Ic], axis=-1)  # [batch, 96, 128, 2]
        output_G2 = self.model(input_G2)  # [batch, 96, 128, 1] dtype=float32
        output_G2 = tf.cast(output_G2, dtype=tf.float16)
        return output_G2

    # LOSS
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

        poseMaskLoss = self.PoseMaskloss(I_PT2, It, Mt)

        loss = gen_cost + poseMaskLoss * 10

        return loss

    # METRICHE
    def ssim(self, I_PT2, It, mean_0, mean_1, unprocess_function):
        It = tf.reshape(It, [-1, 96, 128, 1])
        I_PT2 = tf.reshape(I_PT2, [-1, 96, 128, 1])

        It = tf.cast(unprocess_function(It, mean_1), dtype=tf.uint16)
        I_PT2 = tf.cast(unprocess_function(I_PT2, mean_0), dtype=tf.uint16)

        result = tf.image.ssim(I_PT2, It, max_val=tf.reduce_max(It) - tf.reduce_min(It))
        mean = tf.reduce_mean(result)

        return mean


    def mask_ssim(self, I_PT2, It, Mt, mean_0, mean_1, unprocess_function):
        It = tf.reshape(It, [-1, 96, 128, 1])
        Mt = tf.reshape(Mt, [-1, 96, 128, 1])
        I_PT2 = tf.reshape(I_PT2, [-1, 96, 128, 1])

        It = tf.cast(unprocess_function(It, mean_1, 32765.5), dtype=tf.uint16)
        I_PT2 = tf.cast(unprocess_function(I_PT2, mean_0, 32765.5), dtype=tf.uint16)
        Mt = tf.cast(Mt, dtype=tf.uint16)

        mask_image_raw_1 = Mt * It
        mask_refined_result = Mt * I_PT2

        result = tf.image.ssim(mask_image_raw_1, mask_refined_result, max_val=tf.reduce_max(It) - tf.reduce_min(It))
        mean = tf.reduce_mean(result)

        return mean