import math
import numpy as np
from ops import *
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
#from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow_addons.layers import InstanceNormalization as InstanceNormalization
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization, Activation, MaxPooling2D,
    UpSampling2D, Conv2D, SeparableConv2D, Concatenate, BatchNormalization, Flatten, GlobalAveragePooling2D
)

kr=keras.regularizers.l2(0.001)

class Unet(tf.keras.Model):
    def __init__(self, input_shape, channel, output_channel, name, relord_path=None, use_skip=True):
        super(Unet, self).__init__(name=name)

        self.inputs = tf.keras.layers.Input(input_shape, name=name+'_input')
        self.output_channel = output_channel
        self.channel = channel
        self.use_skip = use_skip
        self.network_name = name

        self.model = self.architecture()
        x = self.model([self.inputs])
        self.model = tf.keras.Model([self.inputs], [x], name=self.network_name)
        self.params_num = self.model.count_params()
        self.model.summary()
        if relord_path !=None:
            self.model.load_weights(relord_path)

    def architecture(self):
        skip=[]

        x = Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_regularizer=kr)(self.inputs)
        x = InstanceNormalization()(x)

        for i in range(4):
            skip.append(x) if self.use_skip else skip.append(x.shape[-1])
            self.channel = self.channel * 2 if self.channel<256 else 256
            x = Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_regularizer=kr)(x)
            x = InstanceNormalization()(x)
            x = Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_regularizer=kr)(x)
            x = InstanceNormalization()(x)
            x = MaxPooling2D(pool_size=(2,2))(x)
            
        for i in range(4):
            x = UpSampling2D(size=(2, 2))(x)
            self.channel = skip[-(i+1)].shape[-1] if self.use_skip else skip[-(i+1)]
            x = tf.concat([x, skip[-(i+1)]], axis=-1) if self.use_skip else x
            x = Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_regularizer=kr)(x)
            x = InstanceNormalization()(x)
            x = Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_regularizer=kr)(x)
            x = InstanceNormalization()(x)
        
        x = Conv2D(filters=self.output_channel, kernel_size=3, padding='same', kernel_regularizer=kr)(x)
        
        return tf.keras.Model(self.inputs, x)

    @tf.function
    def call(self, inputs):

        x = self.model([inputs])

        return x

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, norm='in', separable=False):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.separable = separable
        self.conv1 = SeparableConv2D(filters=self.channels, kernel_size=3, padding='same', activation=tf.nn.relu) if self.separable \
                     else Conv2D(filters=self.channels, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv2 = SeparableConv2D(filters=self.channels, kernel_size=3, padding='same', activation=tf.nn.relu) if self.separable \
                     else Conv2D(filters=self.channels, kernel_size=3, padding='same', activation=tf.nn.relu)
        if norm=='bn':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
        if norm=='in':
            self.norm1 = InstanceNormalization()
            self.norm2 = InstanceNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + inputs
        return x
        
class AE(tf.keras.Model):
    def __init__(self, input_shape, channel, output_channel, n_res=3, separable=False, name='ae'):
        super(AE, self).__init__(name=name)

        self.inputs = tf.keras.layers.Input(input_shape, name=name+'_input')
        self.output_channel = output_channel
        self.channel = channel
        self.n_res = n_res
        self.separable = separable
        self.network_name = name
        self.down_conv = tf.keras.Sequential(
            [
                Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu),
                InstanceNormalization(),
                SeparableConv2D(filters=self.channel*2, kernel_size=3, padding='same', activation=tf.nn.relu) if self.separable \
                    else Conv2D(filters=self.channel*2, kernel_size=3, padding='same', activation=tf.nn.relu),
                MaxPooling2D(pool_size=(2,2)),
                InstanceNormalization(),
                SeparableConv2D(filters=self.channel*4, kernel_size=3, padding='same', activation=tf.nn.relu) if self.separable \
                    else Conv2D(filters=self.channel*4, kernel_size=3, padding='same', activation=tf.nn.relu),
                MaxPooling2D(pool_size=(2,2)),
                InstanceNormalization(),
            ])
        self.res_layers = [
            ResBlock(self.channel*4, separable=self.separable)
            for _ in range(self.n_res)
        ]
        self.up_conv = tf.keras.Sequential(
            [
                UpSampling2D(size=(2, 2)),
                SeparableConv2D(filters=self.channel*2, kernel_size=3, padding='same', activation=tf.nn.relu) if self.separable \
                    else Conv2D(filters=self.channel*2, kernel_size=3, padding='same', activation=tf.nn.relu),
                InstanceNormalization(),
                UpSampling2D(size=(2, 2)),
                SeparableConv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu) if self.separable \
                    else Conv2D(filters=self.channel, kernel_size=3, padding='same', activation=tf.nn.relu),
                InstanceNormalization(),
                Conv2D(filters=self.output_channel, kernel_size=3, padding='same'),
            ])

        self.model = self.architecture()
        x = self.model([self.inputs])
        self.model = tf.keras.Model([self.inputs], [x], name=self.network_name)
        self.params_num = self.model.count_params()
        self.model.summary()

    def architecture(self):

        x = self.down_conv(self.inputs)
        for layer in self.res_layers:
            x = layer(x)
        x = self.up_conv(x)
        
        return tf.keras.Model(self.inputs, x)

    @tf.function
    def call(self, inputs):

        x = self.model([inputs])

        return x

class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, channel, label_size, sn=False, name='dis'):
        super(Discriminator, self).__init__(name=name)

        self.inputs = tf.keras.layers.Input(input_shape, name='d_input')
        self.channel = channel
        self.label_size = label_size
        self.sn = sn
        self.network_name = name
        self.backbone = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D((1,2)),#25,23->27,27->13,13
                SpectralNormalization(Conv2D(filters=self.channel, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, kernel_regularizer=kr)) if self.sn\
                    else Conv2D(filters=self.channel, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, kernel_regularizer=kr),
                tf.keras.layers.ZeroPadding2D((1,1)),#13,13->15,15->7,7
                SpectralNormalization(Conv2D(filters=self.channel*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, kernel_regularizer=kr)) if self.sn\
                    else Conv2D(filters=self.channel*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, kernel_regularizer=kr),
                tf.keras.layers.ZeroPadding2D((1,1)),#7,7->9,9->4,4
                SpectralNormalization(Conv2D(filters=self.channel*4, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, kernel_regularizer=kr)) if self.sn\
                    else Conv2D(filters=self.channel*4, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, kernel_regularizer=kr),
                #4,4->2,2
                SpectralNormalization(Conv2D(filters=self.channel*4, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, kernel_regularizer=kr)) if self.sn\
                    else Conv2D(filters=self.channel*4, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, kernel_regularizer=kr),
            ])
        self.logit_head = tf.keras.Sequential(
            [
                Flatten(),
                SpectralNormalization(Dense(256, activation=tf.nn.leaky_relu, kernel_regularizer=kr)) if self.sn\
                    else Dense(256, activation=tf.nn.leaky_relu, kernel_regularizer=kr),
                SpectralNormalization(Dense(64, activation=tf.nn.leaky_relu, kernel_regularizer=kr)) if self.sn\
                    else Dense(64, activation=tf.nn.leaky_relu, kernel_regularizer=kr),
                SpectralNormalization(Dense(1, kernel_regularizer=kr)) if self.sn\
                    else Dense(1, kernel_regularizer=kr),
            ])
        self.label_head = tf.keras.Sequential(
            [
                GlobalAveragePooling2D(),
                Dense(64, activation=tf.nn.relu, kernel_regularizer=kr),
                Dense(self.label_size, kernel_regularizer=kr),
            ])

        self.arch = self.architecture()
        logit, label = self.arch(self.inputs)
        self.model = tf.keras.Model([self.inputs], [logit, label], name=self.network_name)
        self.model.summary()
        self.model.count_params()

    def architecture(self):
        x = self.backbone(self.inputs)
        logit = self.logit_head(x)
        label = self.label_head(x)

        return tf.keras.Model([self.inputs], [logit, label])

    @tf.function
    def call(self, inputs):

        logit, label = self.model(inputs)

        return logit, label
